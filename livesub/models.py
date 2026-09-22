"""Model loading and inference wrappers.

ASR: faster-whisper (CTranslate2), with a local alias table for the
anime-tuned model. MT: Sakura-7B (llama.cpp GGUF, CPU or partial GPU),
NLLB-200-600M (CT2, GPU), or Qwen2.5-1.5B (transformers, GPU).
Every heavy library is imported lazily inside __init__/load_model so that
importing this module (and the package) stays cheap and side-effect free.

All MT backends share one shallow interface: translate(text) -> str plus a
last_stats dict the pipeline merges into the JSONL trace. The pipeline only
ever sees that interface — backend choice is a constructor detail.
"""
import os
import re
import sys
import time

from livesub.config import HAVE_CUDA_DLLS, ROOT

# 本地 CT2 模型的别名表：--model anime 即加载这个目录。
# quantumcookie/anime-whisper-ct2 是 litagin/anime-whisper（756M）的 CTranslate2 版，
# 底座 kotoba-whisper-v2.0，5300h 动漫语音微调；官方特性：抗幻觉、识别非言语声
# （笑/叹气/呼吸）、支持 NSFW 音频。日语 ASMR 场景比 Whisper small 对口得多。
# 注意：官方要求不要加 initial_prompt，会掉点。
LOCAL_ASR_MODELS = {
    "anime": ROOT / "models" / "anime-whisper-ct2",
}


def load_model(args):
    from faster_whisper import WhisperModel
    device = args.device
    if device == "cuda" and not HAVE_CUDA_DLLS:
        print("[warn] cuBLAS/cuDNN runtime not found, using CPU", file=sys.stderr)
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    # 先查别名表：命中的话直接走本地目录，不联网
    src = LOCAL_ASR_MODELS.get(args.model)
    if src is not None:
        if not (src / "model.bin").exists():
            raise FileNotFoundError(
                f"本地模型 {args.model} 不完整：缺少 {src / 'model.bin'}")
        print(f"[asr] 本地 CT2 模型 {args.model} -> {src}")
        return WhisperModel(str(src), device=device, compute_type=compute,
                            local_files_only=True)
    return WhisperModel(args.model, device=device, compute_type=compute,
                        download_root=str(ROOT / "models"))

class SakuraMT:
    """Sakura-7B Q4 on CPU/RAM. GPU stays with Whisper."""
    SYS = ("你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
           "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。")
    GGUF = ROOT / "models" / "sakura" / "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"

    def __init__(self, n_gpu_layers=0):
        from llama_cpp import Llama
        if not self.GGUF.exists():
            raise FileNotFoundError(f"missing {self.GGUF}")
        n_threads = max(4, min(12, (os.cpu_count() or 8) - 2))
        t0 = time.time()
        self.llm = Llama(
            model_path=str(self.GGUF),
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print(f"[mt] Sakura-7B IQ4XS 已加载 (cpu threads={n_threads} ngl={n_gpu_layers} {time.time()-t0:.1f}s)")
        self.last_stats = {"load_s": round(time.time() - t0, 3), "threads": n_threads, "ngl": n_gpu_layers}

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            self.last_stats = {}
            return ""
        prompt = (
            f"<|im_start|>system\n{self.SYS}<|im_end|>\n"
            f"<|im_start|>user\n将下面的日文文本翻译成中文：{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        t0 = time.time()
        max_tokens = min(80, max(12, len(text) * 3 + 8))
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.3,
            repeat_penalty=1.05,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        zh = (out["choices"][0]["text"] or "").strip()
        usage = out.get("usage") or {}
        tim = out.get("timings") or {}
        self.last_stats = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "pred_ms": tim.get("predicted_ms"),
            "prompt_ms": tim.get("prompt_ms"),
            "pred_per_s": tim.get("predicted_per_second"),
            "gen_s": round(time.time() - t0, 3),
            "in_chars": len(text),
            "out_chars": len(zh),
            "max_tokens": max_tokens,
        }
        return zh

class NllbMT:
    """Dedicated ja→zh MT (NLLB-200 distilled 600M). No chat prompt, no safety filter."""
    MODEL_ID = "facebook/nllb-200-distilled-600M"
    SRC = "jpn_Jpan"
    TGT = "zho_Hans"

    def __init__(self):
        import ctranslate2
        from transformers import AutoTokenizer
        ct2_dir = ROOT / "models" / "nllb"
        if not (ct2_dir / "model.bin").exists():
            raise FileNotFoundError(f"missing {ct2_dir / 'model.bin'}; convert first")
        cache = str(ROOT / "models")
        self.tok = AutoTokenizer.from_pretrained(
            self.MODEL_ID, src_lang=self.SRC, cache_dir=cache)
        self.tr = ctranslate2.Translator(
            str(ct2_dir), device="cuda", compute_type="float16")
        print("[mt] NLLB-200-600M 已加载 (cuda fp16, ja→zh)")
        self.last_stats = {}

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            return ""
        source = self.tok.convert_ids_to_tokens(self.tok.encode(text))
        max_len = min(160, max(12, len(source) * 3))
        results = self.tr.translate_batch(
            [source], target_prefix=[[self.TGT]],
            beam_size=1, repetition_penalty=1.3,
            no_repeat_ngram_size=3, disable_unk=True,
            max_decoding_length=max_len)
        hyp = results[0].hypotheses[0]
        if hyp and hyp[0] == self.TGT:
            hyp = hyp[1:]
        ids = self.tok.convert_tokens_to_ids(hyp)
        out = self.tok.decode(ids, skip_special_tokens=True).strip()
        core = re.sub(r"[\s,，.。、!！?？~]+", "", out)
        if len(core) >= 8 and len(set(core)) <= 2:
            return ""
        return out

class QwenMT:
    SYS = ("你是日中字幕翻译。把用户输入的日语文本逐句直译成简体中文："
           "不得增删原文没有的内容；拟声词、喘息、语气词按原样用中文对应字保留；"
           "粗俗或暧昧用语直译，不要美化或改写；无法确定含义的词按字面音译。"
           "只输出译文。")

    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, device_map="cuda")
        self.model.eval()
        print("[mt] Qwen2.5-1.5B 已加载 (cuda fp16)")
        self.last_stats = {}

    def translate(self, text):
        msgs = [{"role": "system", "content": self.SYS},
                {"role": "user", "content": text}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=160, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

def load_translator(args):
    if args.mt == "none":
        return None
    if args.mt == "sakura":
        try:
            return SakuraMT(n_gpu_layers=getattr(args, "mt_ngl", 0))
        except Exception as e:
            print(f"[mt-warn] Sakura 加载失败({e})，只显示日语原文", file=sys.stderr)
            return None
    if args.mt == "qwen":
        path = ROOT / "models" / "qwen"
        if not (path / "config.json").exists():
            print("[mt] 未找到 models/qwen，只显示日语原文", file=sys.stderr)
            return None
        try:
            return QwenMT(str(path))
        except Exception as e:
            print(f"[mt-warn] Qwen 加载失败({e})，只显示日语原文", file=sys.stderr)
            return None
    try:
        return NllbMT()
    except Exception as e:
        print(f"[mt-warn] NLLB 加载失败({e})，只显示日语原文", file=sys.stderr)
        return None
