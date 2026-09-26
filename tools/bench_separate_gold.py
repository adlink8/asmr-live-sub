"""分离器当转写辅助的实测：金标窗上对比"原声 vs 分离后人声"的转写 CER。

用户构想：混合音频里的音效干扰识别 → 分离器剥出人声 → 转写更稳 → 当 teacher 标签。
本实验直接测这个前提：Demucs 被依赖问题排除，用 audio-separator（MDX/roformer）。
同引擎同参数转写两版音频，与官方台本（26 段真值）比 CER。

    .venv\\Scripts\\python.exe tools/bench_separate_gold.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import live_sub  # noqa: F401  — CUDA DLL 注册副作用

from bench_large_v3_local import cer, align_and_score  # 同口径

WAV = ROOT / "benchmarks" / "scene_random_talk.wav"
GT = ROOT / "dataset" / "scene_random_talk.gt_ja.json"
ASR_DIR = ROOT / "models" / "anime-whisper-ct2"
SEP_OUT = ROOT / "logs" / "sep_out"


def transcribe(path):
    """与 livesub 线上 asr_decode 完全同参（beam1/关上文条件化）——
    beam5+默认 condition_on_previous_text 会在长音频触发重复循环幻觉，
    两边一起烂，比不出分离的净效果。"""
    from faster_whisper import WhisperModel
    m = WhisperModel(str(ASR_DIR), device="cuda", compute_type="float16")
    segs, info = m.transcribe(str(path), language="ja", task="transcribe",
                              beam_size=1, condition_on_previous_text=False,
                              no_speech_threshold=0.6, patience=2,
                              vad_filter=False)
    return [(s.start, s.end, s.text) for s in segs]


def main():
    gt = json.loads(GT.read_text(encoding="utf-8"))
    gt = gt if isinstance(gt, list) else gt.get("segments", [])

    cached = sorted(SEP_OUT.glob("*Vocals*.wav"))
    if cached:
        vocals = cached[0]
        print(f"[1/4] 复用已有分离产物: {vocals.name}")
    else:
        print("[1/4] audio-separator 分离人声（模型首次跑需下载）...")
        from audio_separator.separator import Separator
        SEP_OUT.mkdir(parents=True, exist_ok=True)
        sep = Separator(output_dir=str(SEP_OUT), output_format="wav",
                        model_file_dir=str(ROOT / "logs" / "sep_models"))
        t0 = time.time()
        sep.load_model()  # 默认 bs_roformer（人声分离 SOTA，torch+CUDA 推理）
        print(f"      模型就绪 {time.time()-t0:.0f}s")
        t0 = time.time()
        outs = sep.separate(str(WAV))
        print(f"      分离完成 {time.time()-t0:.0f}s -> {outs}")
        vocals = next((SEP_OUT / o for o in outs
                       if "vocals" in o.lower() or "vocal" in o.lower()), None)
        if vocals is None:
            print("      [warn] 未识别 vocals 文件名，取第一个产物", outs)
            vocals = SEP_OUT / outs[0]

    print("[2/4] 转写原声...")
    spans_orig = transcribe(WAV)
    print(f"      {len(spans_orig)} spans")

    print("[3/4] 转写分离后的人声...")
    spans_voc = transcribe(vocals)
    print(f"      {len(spans_voc)} spans")

    print("[4/4] 对齐官方 26 段真值算 CER ...")
    ref_all = "".join(s["official_ja"] for s in gt)
    orig_all = "".join(t for _, _, t in spans_orig)
    voc_all = "".join(t for _, _, t in spans_voc)

    def summ(name, spans):
        per = align_and_score(gt, spans)
        whole = cer(ref_all, "".join(t for _, _, t in spans))
        ok = sorted(per)
        print(f"  {name:12s} 整窗 CER {whole*100:5.1f}% | "
              f"段级中位 {ok[len(ok)//2]*100:5.1f}% | "
              f"全对段 {sum(1 for x in per if x == 0)}/26")
        return per, whole

    print("\n=== 结果（越低越好）===")
    per_o, _ = summ("原声", spans_orig)
    per_v, _ = summ("分离后人声", spans_voc)
    print("\n段级明细（原声% / 分离% | 官方句前 16 字）:")
    for s, a, b in zip(gt, per_o, per_v):
        flag = "  <-- 分离更好" if b < a - 0.05 else ("  <-- 分离更差" if a < b - 0.05 else "")
        print(f"  {a*100:5.1f} / {b*100:5.1f} | {s['official_ja'][:16]}{flag}")

    out = ROOT / "logs" / "bench_separate_gold.json"
    out.write_text(json.dumps(
        {"orig_spans": spans_orig, "voc_spans": spans_voc,
         "vocals_file": str(vocals)}, ensure_ascii=False), encoding="utf-8")
    print(f"\n明细: {out}")


if __name__ == "__main__":
    main()
