import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(ROOT))  # 脚本在 tools/ 下，live_sub 兼容层在仓库根
import live_sub

def generate_teacher_gt(audio_path: Path, out_json: Path, model_name: str = "anime",
                        ja_only: bool = False):
    print(f"\n=======================================================", flush=True)
    print(f"[*] 启动 Teacher Model 离线真值提取", flush=True)
    print(f"    输入音频: {audio_path.name}", flush=True)
    print(f"    模型模式: 离线全篇 (Full-Pass) + Beam Search (beam_size=5)", flush=True)
    
    class Args:
        model = model_name
        device = "cuda"
        
    t0 = time.time()
    model = live_sub.load_model(Args)
    print(f"    ASR 模型加载完成: {time.time()-t0:.2f}s", flush=True)
    
    t1 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language="ja",
        beam_size=5,
        condition_on_previous_text=True,
        vad_filter=False,
        word_timestamps=False
    )
    
    raw_segments = []
    for s in segments:
        raw_segments.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 3),
            "no_speech_prob": round(s.no_speech_prob, 3),
        })
    asr_cost = time.time() - t1
    print(f"    ASR 全篇提取完成: {len(raw_segments)} 段, 耗时 {asr_cost:.2f}s", flush=True)
    
    # Sakura 离线翻译（--ja-only 时跳过：参考译文统一走
    # build_quality_reference.py 的贪心精译，这里旧设置的译文不进数据集）
    gt_entries = []
    translator = None
    if not ja_only:
        print("[*] 启动离线真值翻译 (Sakura-7B 上下文优化)", flush=True)
        translator = live_sub.SakuraMT(n_gpu_layers=0)

    for i, seg in enumerate(raw_segments, 1):
        ja = seg["text"]
        zh = translator.translate(ja) if translator else ""
        # 解码退化标记：含 U+FFFD 替换符 = CT2 明确吐出的乱码段
        # （全篇无 VAD 解码在多人重叠/快语速区会幻觉出垃圾，拿它当真值
        #  会得出误导性坏数字，必须剔除并记入覆盖率）
        degraded = "\ufffd" in ja
        gt_entries.append({
            "idx": i,
            "t0": seg["start"],
            "t1": seg["end"],
            "ja": ja,
            "zh": zh,
            "avg_logprob": seg["avg_logprob"],
            "no_speech_prob": seg["no_speech_prob"],
            "excluded": degraded,
            "exclude_reason": "decode_garbage" if degraded else "",
        })
        print(f"  [{i:02d}] {seg['start']:.1f}s~{seg['end']:.1f}s: {ja[:60]}"
              + (f" -> {zh[:40]}" if zh else "")
              + ("  [剔除: 解码乱码]" if degraded else ""), flush=True)

    full_data = {
        "audio": audio_path.name,
        "audio_path": str(audio_path),
        "duration_s": round(raw_segments[-1]["end"] if raw_segments else 0.0, 2),
        "total_segments": len(gt_entries),
        "full_ja": "".join(e["ja"] for e in gt_entries),
        "segments": gt_entries
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    print(f"[✓] 黄金真值基准成功保存至: {out_json}")
    return full_data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to wav file")
    ap.add_argument("--out", required=True, help="Output json path")
    ap.add_argument("--ja-only", action="store_true",
                    help="只产日文真值，跳过 Sakura 翻译（参考译文走 build_quality_reference.py）")
    args = ap.parse_args()
    
    audio_p = Path(args.audio)
    out_p = Path(args.out)
    generate_teacher_gt(audio_p, out_p, ja_only=args.ja_only)

if __name__ == "__main__":
    main()
