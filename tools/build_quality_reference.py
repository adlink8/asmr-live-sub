"""质量参考译文构建（离线，无延迟约束）。

实时管线的翻译受三重约束：5s 切块无上下文、max_tokens<=80 截断、
temperature=0.1 采样。参考侧没有这些约束——同一个 Sakura 模型、同一套
prompt 模板，只把推理配置换成贪心解码 + 长输出 + GPU 全 offload
（本脚本不加载 Whisper，显存全给 MT）。

注意不带前文上下文：实测"参考上文"格式下该量化模型会把上文一并译出、
真实译文混在"文本："标记之后，解析不可靠。上下文不是那三条实时约束之一，
为它引入噪声不划算，参考译文因此只衡量"推理配置"的差距。

每段产出两个译文：
  zh      官方台本日文 -> 离线精译。"完美识别 + 离线精译"的上限，
          evaluate_accuracy 的 --mt-ref 消费它，测端到端总差距。
  zh_live 实时日文 -> 同一套离线精译。把总差距拆成"识别损伤 + 翻译退化"。

输入：官方台本 txt + 一次实时运行的 jsonl（只用 seg_id/t_start/t_end 做时间轴）。
输出：json，含 provenance 与 excluded 标记（拟声词/低置信匹配段不译不评）。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# 先注册 CUDA DLL 搜索路径，否则 llama_cpp 找不到 ggml-cuda 的依赖
import livesub.config  # noqa: F401
from align_official_script import align_script_to_audio

SYS = ("你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
       "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。")
GGUF = ROOT / "models" / "sakura" / "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"
IM_START = "<" + "|im_start|>"
IM_END = "<" + "|im_end|>"


def build_prompt(ja):
    return (IM_START + "system\n" + SYS + IM_END + "\n" +
            IM_START + "user\n将下面的日文文本翻译成中文：" + ja + IM_END + "\n" +
            IM_START + "assistant\n")


class QualityTranslator:
    """Sakura-7B 离线精译：贪心解码、长输出、GPU 全 offload。

    与实时 SakuraMT 的差别只有推理参数（temperature=0 / max_tokens=512 /
    repeat_penalty=1.12）和显存策略，模型与 prompt 模板同源，
    所以参考译文衡量的是"推理配置"而非"另一个模型"的差距。
    """

    def __init__(self, n_gpu_layers=99, n_ctx=4096):
        from llama_cpp import Llama
        if not GGUF.exists():
            raise FileNotFoundError(f"missing {GGUF}")
        t0 = time.time()
        self.llm = Llama(
            model_path=str(GGUF),
            n_ctx=n_ctx,
            n_threads=max(4, min(12, (os.cpu_count() or 8) - 2)),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print(f"[mt] Sakura-7B 离线模式已加载 (ngl={n_gpu_layers} ctx={n_ctx} "
              f"{time.time() - t0:.1f}s)")

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            return ""
        out = self.llm(
            build_prompt(text),
            max_tokens=512,
            temperature=0.0,
            top_k=1,
            repeat_penalty=1.12,
            stop=[IM_END, "<" + "|endoftext|>"],
        )
        return (out["choices"][0]["text"] or "").strip()


def _load_gt_segments(gt_json_path):
    """从真值 json 读段（兼容 align 的 official_ja 与 teacher 的 ja 两种字段）。

    gt 模式没有 live 输入，zh_live 不产（它是按次运行的分析产物，不属于数据集）。
    """
    d = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))
    segments = []
    for i, s in enumerate(d.get("segments", [])):
        ja = (s.get("official_ja") or s.get("ja") or "").strip()
        segments.append({
            "idx": i + 1,
            "seg_id": s.get("seg_id", s.get("idx", i)),
            "t0": s.get("t0", 0.0),
            "t1": s.get("t1", 0.0),
            "official_ja": ja,
            "live_ja": "",
            "match_score": s.get("match_score"),
            "excluded": bool(s.get("excluded", False)),
            "exclude_reason": s.get("exclude_reason", ""),
        })
    return {"script_file": None, "audio_file": d.get("audio"),
            "segments": segments}


def build_reference(script_path, live_jsonl_path, out_path, limit=None,
                    gt_json_path=None):
    if gt_json_path:
        # gt 模式：真值已存在（teacher 级或已对齐的金级），直接精译。
        # 没有 live 输入，zh_live 不产（它是按次运行的分析产物，不属于数据集）。
        result = _load_gt_segments(gt_json_path)
        gt_source = "gt_json"
        with_live = False
    else:
        # 脚本模式：先落盘对齐结果，即使翻译中途崩溃时间轴/剔除判定也已保留可查
        result = align_script_to_audio(script_path, live_jsonl_path, out_path)
        gt_source = "official_script"
        with_live = True
    segments = result["segments"]
    if limit:
        segments = segments[:limit]

    translator = QualityTranslator()
    t0 = time.time()
    for i, seg in enumerate(segments):
        if seg["excluded"]:
            seg["zh"] = ""
            seg["zh_live"] = ""
            print(f"[{i + 1}/{len(segments)}] seg={seg['seg_id']} "
                  f"跳过（{seg['exclude_reason']}）")
            continue
        seg["zh"] = translator.translate(seg["official_ja"])
        seg["zh_live"] = (translator.translate(seg["live_ja"])
                          if with_live else "")
        print(f"[{i + 1}/{len(segments)}] seg={seg['seg_id']} "
              f"zh={seg['zh'][:24]}"
              + (f" | zh_live={seg['zh_live'][:24]}" if seg["zh_live"] else ""))

    translated = sum(1 for s in segments if s["zh"])
    out = {
        "script_file": result["script_file"],
        "audio_file": result["audio_file"],
        "total_segments": len(segments),
        "provenance": {
            "model": "sakura-7b-qwen2.5-v1.0-iq4xs (gguf iq4xs)",
            "decode": "greedy temperature=0 top_k=1",
            "max_tokens": 512,
            "repeat_penalty": 1.12,
            "n_ctx": 4096,
            "context": "none (same prompt template as live)",
            "gt_source": gt_source,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_s": round(time.time() - t0, 1),
            "translated": translated,
            "excluded": sum(1 for s in segments if s["excluded"]),
        },
        "segments": segments,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[OK] 质量参考译文: {out_path}")
    print(f"     段数={len(segments)} 已译={translated} "
          f"剔除={out['provenance']['excluded']} "
          f"耗时={out['provenance']['build_s']}s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--script", default=None, help="官方台本 txt（脚本模式）")
    ap.add_argument("--live", default=None, help="一次实时运行的 jsonl（脚本模式）")
    ap.add_argument("--gt", default=None,
                    help="真值 json（gt 模式：直接精译真值日文，不联网对齐）")
    ap.add_argument("--out", required=True, help="输出参考译文 json")
    ap.add_argument("--limit", type=int, default=None,
                    help="只译前 N 段（冒烟用）")
    args = ap.parse_args()
    if not args.gt and not (args.script and args.live):
        ap.error("要么 --gt <真值json>，要么 --script <台本txt> --live <jsonl>")
    build_reference(Path(args.script) if args.script else None,
                    Path(args.live) if args.live else None,
                    Path(args.out), limit=args.limit,
                    gt_json_path=Path(args.gt) if args.gt else None)


if __name__ == "__main__":
    main()
