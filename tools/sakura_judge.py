"""Sakura-7B 本地语义裁判（三票制中的第三票）。

用法:
  python tools/sakura_judge.py --input dataset/scene_xxx.judge_input.json [--ngl 99]
  python tools/sakura_judge.py --input "dataset/scene_asmr1*_*.judge_input.json"   # 支持 glob

输入  : judge_input.json（pairs[].ja / sakura_live / human）
输出  : 与 semantic_judge.json 逐字段同构的 <scene>.semantic_judge_sakura.json
判分  : correct / partial / wrong + kind(dialogue|interception)，temperature=0
说明  : 弱裁判，只当第三票/分歧参考，不作为权威；JSON 解析失败的记 unparsed 并剔除出百分比。
"""
import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models" / "sakura" / "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"

SYS = (
    "你是日中翻译质量裁判。给定：日文原文（仲裁依据）、人工参考中文、待判中文译文。"
    "判定规则：意思正确=correct；意思基本对但有遗漏/偏差/明显风格问题=partial；意思错误=wrong。"
    "若内容是非语义的拟声词/纯计数（如「啾噜噜」「3、2、1」），kind=interception，verdict 按意思符合程度判；"
    "否则 kind=dialogue。只输出一个 JSON，不要任何其他文字："
    '{"verdict":"correct|partial|wrong","kind":"dialogue|interception","note":"不超过15字理由"}'
)


def chat(llm, user, max_tokens=512):
    prompt = (
        f"<|im_start|>system\n{SYS}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    out = llm(prompt, max_tokens=max_tokens, temperature=0.0, top_p=0.1,
              stop=["<|im_end|>", "<|endoftext|>"])
    return (out["choices"][0]["text"] or "").strip()


def parse_verdict(text):
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        if v.get("verdict") in ("correct", "partial", "wrong") and v.get("kind") in ("dialogue", "interception"):
            return v
    except Exception:
        pass
    return None


def judge_file(llm, path, ngl, model_name):
    d = json.load(open(path, encoding="utf-8"))
    scene = d.get("scene") or Path(path).stem.replace(".judge_input", "")
    pairs = d.get("pairs", [])
    verdicts = []
    unparsed = 0
    t0 = time.time()
    for i, p in enumerate(pairs):
        user = f"日文原文：{p.get('ja','')}\n人工参考：{p.get('human','')}\n待判译文：{p.get('sakura_live','')}"
        raw = chat(llm, user)
        v = parse_verdict(raw)
        if v is None:
            raw = chat(llm, user + "\n（再次提醒：只输出 JSON，格式为 {\"verdict\":\"correct|partial|wrong\",\"kind\":\"dialogue|interception\",\"note\":\"...\"}）")
            v = parse_verdict(raw)
        if v is None:
            unparsed += 1
            verdicts.append({"seg_id": p.get("seg_id"), "sakura": "unparsed", "kind": "dialogue", "note": raw[:60]})
        else:
            verdicts.append({"seg_id": p.get("seg_id"), "sakura": v["verdict"], "kind": v["kind"], "note": v.get("note", "")})
        if (i + 1) % 10 == 0:
            print(f"  [{scene}] {i+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)

    ok = [v for v in verdicts if v["sakura"] != "unparsed"]
    dlg = [v for v in ok if v["kind"] == "dialogue"]
    itc = [v for v in ok if v["kind"] == "interception"]

    def tally(vs):
        c = sum(1 for v in vs if v["sakura"] == "correct")
        pa = sum(1 for v in vs if v["sakura"] == "partial")
        w = sum(1 for v in vs if v["sakura"] == "wrong")
        n = len(vs)
        return {"correct": c, "partial": pa, "wrong": w, "n": n,
                "correctPct": round(100 * c / n, 1) if n else 0.0}

    inter = {}
    for v in itc:
        inter.setdefault(v["kind"], {"n": 0, "correct": 0, "partial": 0, "wrong": 0})
        inter[v["kind"]]["n"] += 1
        inter[v["kind"]][v["sakura"]] += 1

    return {
        "scene": scene,
        "method": ("LLM semantic judge (%s local judge mode, ngl=%d), meaning-only, "
                   "JA as ambiguity arbiter. 第三票/弱裁判：仅用于交叉验证与分歧参考，非权威。" % (model_name, ngl)),
        "built_at": time.strftime("%Y-%m-%d"),
        "total_pairs": len(pairs),
        "summary": {"sakura_live": tally(ok)},
        "summary_dialogue": {"sakura_live": tally(dlg)},
        "interception": inter,
        "interception_note": "拟声/纯计数 cue 单列，不入语义分母",
        "unparsed": unparsed,
        "verdicts": verdicts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="judge_input.json 路径或 glob")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--model", default=str(GGUF), help="GGUF 路径（默认 Sakura-7B；也可用 instruct 模型）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认与输入同目录）")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        print(f"[error] 无匹配文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    import os
    free_mb = 0
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        total, used = map(int, r.stdout.strip().split(", "))
        free_mb = total - used
    except Exception:
        pass
    if free_mb and free_mb < 5000:
        print(f"[error] GPU 空闲仅 {free_mb}MiB（<5000），先腾显存再跑", file=sys.stderr)
        sys.exit(2)

    # CUDA DLL 路径注册是 livesub.config 的 import 副作用，必须在 llama_cpp 之前
    import livesub.config  # noqa: F401  (_add_cuda_dlls)
    from llama_cpp import Llama
    model_path = args.model
    model_name = Path(model_path).stem
    print(f"[load] {model_name} ngl={args.ngl} ...", flush=True)
    llm = Llama(model_path=model_path, n_gpu_layers=args.ngl, n_ctx=2048, verbose=False)

    for f in files:
        print(f"[judge] {f}", flush=True)
        result = judge_file(llm, f, args.ngl, model_name)
        outdir = Path(args.outdir) if args.outdir else Path(f).parent
        out = outdir / (result["scene"] + ".semantic_judge_sakura.json")
        with open(out, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=1)
        s = result["summary_dialogue"]["sakura_live"]
        print(f"  -> {out.name}  dialogue correct {s['correct']}/{s['n']} = {s['correctPct']}%  (unparsed {result['unparsed']})", flush=True)


if __name__ == "__main__":
    main()
