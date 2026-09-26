"""large-v3 本地 int8 转写质量实测：gold 窗（官方台本真值）上与现役 anime-whisper 对比。

真值：dataset/scene_random_talk.gt_ja.json（Blue Archive 官方台本对齐，26 段，
含 live_ja=现役模型同窗转写）。参照基线：tools/asr_checkup.py 的 cer。
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import live_sub  # noqa: F401,E402  — 导入副作用：注册 CUDA DLL 目录（cublas/cudnn）

WAV = ROOT / "benchmarks" / "scene_random_talk.wav"
GT = ROOT / "dataset" / "scene_random_talk.gt_ja.json"


def cer(ref: str, hyp: str) -> float:
    """字符编辑距离率（与 asr_checkup 同口径）。"""
    import unicodedata

    def norm(t):
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"[、。！？…「」『』（）\s　，,.!?\-—ー①-⑳★☆♪～〜]", "", t)
        return t

    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def align_and_score(segs, text_spans):
    """把转写输出按官方段 t0/t1 归位：取与段重叠最长的转写 span 的文本。"""
    scores = []
    for seg in segs:
        t0, t1, ref = seg["t0"], seg["t1"], seg["official_ja"]
        # 该时间区间内覆盖最多的转写文本
        inside = [t for (a, b, t) in text_spans if a < t1 and b > t0]
        hyp = "".join(inside) if inside else ""
        scores.append(cer(ref, hyp))
    return scores


def main():
    segs = json.loads(GT.read_text(encoding="utf-8"))
    segs = segs if isinstance(segs, list) else segs.get("segments", [])

    from faster_whisper import WhisperModel

    t0 = time.time()
    print("[1/3] 加载 large-v3 int8（首次运行需下载 ~1.5GB）...")
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    print(f"      模型就绪 {time.time()-t0:.0f}s")

    print("[2/3] 转写 180s gold 窗（vad_filter 关，保留全部输出便于对齐）...")
    t0 = time.time()
    segments, info = model.transcribe(str(WAV), language="ja", vad_filter=False,
                                      beam_size=5)
    spans = [(s.start, s.end, s.text) for s in segments]
    wall = time.time() - t0
    print(f"      转写完成：{len(spans)} spans，耗时 {wall:.1f}s "
          f"（180s 音频 = {180/wall:.0f}x 实时）")

    print("[3/3] 按官方段对齐算 CER ...")
    lv3 = align_and_score(segs, spans)
    live_scores = [cer(s["official_ja"], s.get("live_ja") or "") for s in segs]

    def summ(name, xs):
        xs_ok = [x for x in xs if x is not None]
        n_perfect = sum(1 for x in xs_ok if x == 0)
        print(f"  {name:22s} CER 中位 {sorted(xs_ok)[len(xs_ok)//2]*100:.1f}% | "
              f"均值 {sum(xs_ok)/len(xs_ok)*100:.1f}% | 全对段 {n_perfect}/{len(xs_ok)}")

    print("\n=== 结果（越低越好）===")
    summ("large-v3 本地 int8", lv3)
    summ("现役 anime-whisper", live_scores)
    # 段级明细
    print("\n段级（CER%: v3 / 现役 | 官方句前 18 字）:")
    for s, a, b in zip(segs, lv3, live_scores):
        print(f"  {a*100:5.1f} / {b*100:5.1f} | {s['official_ja'][:18]}")

    out = ROOT / "logs" / "bench_large_v3_local.json"
    out.write_text(json.dumps(
        {"lv3_cer": lv3, "live_cer": live_scores, "wall_s": wall,
         "spans": spans}, ensure_ascii=False), encoding="utf-8")
    print(f"\n明细: {out}")


if __name__ == "__main__":
    main()
