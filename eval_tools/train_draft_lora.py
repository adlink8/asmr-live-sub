"""Fine-tune Qwen2.5-0.5B-Instruct on Anime/Light Novel translation with LoRA and export to Q4 GGUF.

Workflow:
1. Load Qwen2.5-0.5B-Instruct base weights from local cache.
2. Tokenize 2600 anime parallel samples with Sakura-7B ChatML prompt template.
3. Apply LoRA (r=16, alpha=32) and train for 2 epochs on RTX 4060 GPU with bf16.
4. Merge LoRA weights back into the base model.
5. Convert HuggingFace PyTorch weights to GGUF F16.
6. Quantize GGUF F16 to Q4_K_M -> models/draft/qwen2.5-0.5b-anime-draft-q4.gguf.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

# Register livesub CUDA DLLs and root path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import livesub.config
except Exception as e:
    print(f"[warn] Failed to import livesub.config: {e}", file=sys.stderr)

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL_DIR = (
    ROOT
    / "models"
    / "pretrained_cache"
    / "models"
    / "Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / "master"
)
DATA_PATH = ROOT / "eval_tools" / "data" / "anime_parallel_train.json"
LORA_OUTPUT_DIR = ROOT / "eval_tools" / "lora_output"
MERGED_DIR = ROOT / "eval_tools" / "merged_qwen_05b_anime"
F16_GGUF_PATH = ROOT / "eval_tools" / "qwen2.5-0.5b-anime-draft-f16.gguf"
Q4_GGUF_PATH = ROOT / "models" / "draft" / "qwen2.5-0.5b-anime-draft-q4.gguf"

CONVERT_SCRIPT = Path(
    r"D:\ADLINK\AndroidProjects\Operit\llm\llama\.cxx\operit_deps\llama_cpp-720d7fa4097f76e5d0eade5a92c1df87c1faf9d9-src\convert_hf_to_gguf.py"
)

SAKURA_SYS = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)


class TranslationSFTDataset(Dataset):
    """Custom dataset computing cross-entropy loss exclusively on translation assistant tokens."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 256):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.features = []
        im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        for item in raw_data:
            ja = item["ja"]
            zh = item["zh"]

            prompt_text = (
                f"<|im_start|>system\n{SAKURA_SYS}<|im_end|>\n"
                f"<|im_start|>user\n将下面的日文文本翻译成中文：{ja}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            target_text = f"{zh}<|im_end|>\n"

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)

            input_ids = prompt_ids + target_ids
            # Mask prompt tokens with -100 so model only learns target translation tokens
            labels = [-100] * len(prompt_ids) + target_ids

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            attention_mask = [1] * len(input_ids)

            self.features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        print(f"Loaded {len(self.features)} training samples from {data_path}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def train():
    print("=" * 70)
    print("Step 1: Loading Tokenizer and Base Model (Qwen2.5-0.5B-Instruct)")
    print("=" * 70)

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(f"Base model directory does not exist: {BASE_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print(f"Base model loaded on {model.device}, total params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 70)
    print("Step 2: Configuring LoRA Adapter")
    print("=" * 70)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.enable_input_require_grads()

    print("\n" + "=" * 70)
    print("Step 3: Preparing Dataset & Trainer")
    print("=" * 70)
    train_dataset = TranslationSFTDataset(DATA_PATH, tokenizer, max_length=160)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        logging_steps=15,
        bf16=True,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 70)
    print("Step 4: Executing Fast LoRA Fine-Tuning (2 Epochs)")
    print("=" * 70)
    t_start = time.perf_counter()
    train_result = trainer.train()
    t_train = time.perf_counter() - t_start
    print(f"Training completed in {t_train:.1f} seconds! Final loss: {train_result.training_loss:.4f}")

    print("\n" + "=" * 70)
    print("Step 5: Merging LoRA Weights into Base Model")
    print("=" * 70)
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"Merged model successfully saved to {MERGED_DIR}")

    # Free CUDA memory
    del model
    del merged_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Step 6: Converting Merged PyTorch Model to GGUF F16")
    print("=" * 70)
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(f"llama.cpp conversion script not found at {CONVERT_SCRIPT}")

    cmd_convert = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(MERGED_DIR),
        "--outfile",
        str(F16_GGUF_PATH),
        "--outtype",
        "f16",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CONVERT_SCRIPT.parent) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Running conversion: {' '.join(cmd_convert)}")
    subprocess.run(cmd_convert, check=True, cwd=str(CONVERT_SCRIPT.parent), env=env)
    print(f"GGUF F16 successfully created: {F16_GGUF_PATH} ({F16_GGUF_PATH.stat().st_size / (1024*1024):.1f} MB)")

    print("\n" + "=" * 70)
    print("Step 7: Quantizing GGUF F16 to Q4_K_M (qwen2.5-0.5b-anime-draft-q4.gguf)")
    print("=" * 70)
    import llama_cpp

    Q4_GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M

    print(f"Quantizing {F16_GGUF_PATH} -> {Q4_GGUF_PATH} ...")
    rc = llama_cpp.llama_model_quantize(
        str(F16_GGUF_PATH).encode("utf-8"),
        str(Q4_GGUF_PATH).encode("utf-8"),
        params,
    )
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize failed with exit code: {rc}")

    print(f"Q4 GGUF successfully created: {Q4_GGUF_PATH} ({Q4_GGUF_PATH.stat().st_size / (1024*1024):.1f} MB)")

    # Clean up intermediate F16 GGUF to conserve disk space
    if F16_GGUF_PATH.exists():
        F16_GGUF_PATH.unlink()
        print("Cleaned up intermediate F16 GGUF.")

    print("\n" + "=" * 70)
    print("✅ Fine-tuning, LoRA merging, GGUF export & quantization completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    train()
