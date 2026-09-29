#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script_align_pilot.py — 台本强制对齐试产管线（1627 部双料资产 → 40 部样本）。

流程（全本地，零云依赖）：
  1. download  拉样本作品的日文台本 txt + 全音轨 mp3（复用 asmrone_collect 的 API 封装）
  2. asr       本地 anime-whisper CT2 逐轨听写 → 带时间戳的机器转写段
  3. align     台本逐句 → 与转写段贪心相似度配对 → 钉时间轴
  4. 质检指标  句覆盖率 / 匹配相似度分布 / 时间单调性违规率(钉歪) / 每部汇总
产出：D:/Downloads/asmr-script-align/RJxxx/{script/*.txt, audio/*.mp3, align/*.jsonl}
纪律：R18 内容只出聚合统计，报告不含显性原句。

用法：
  python tools/script_align_pilot.py --stage download --limit 5
  python tools/script_align_pilot.py --stage asr --limit 5
  python tools/script_align_pilot.py --stage align --limit 5
  python tools/script_align_pilot.py --stage report
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from asmrone_collect import api_json, walk_tracks, dedup_audio  # noqa: E402

SAMPLE = Path(r"D:/Downloads/asmr-gold-hunt/script_pilot_sample.json")
WORKDIR = Path(r"D:/Downloads/asmr-script-align")
MODEL_DIR = ROOT / "models" / "anime-whisper-ct2"

JA_KEEP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")   # 含假名/汉字的行才可能是台本正文
SENT_SPLIT = re.compile(r"(?<=[。！？…」』])\s*")
PUNCT = re.compile(r"[\s、。！？：；「」『』（）(),.!?;:\"'…―ー～·，♪♡×○…\-\[\]0-9a-zA-Z]+")
REFUSAL_GUARD = 0  # 占位：无显性文本进入日志


def norm_ja(t: str) -> str:
    t = t or ""
    out = []
    for c in t.strip():
        code = ord(c)
        if 0x30A1 <= code <= 0x30F6:
            c = chr(code - 0x60)
        out.append(c)
    return PUNCT.sub("", "".join(out))


def safe_name(name: str) -> str:
    """Windows 文件名禁字符清洗（半角 ? 等 → 全角）。"""
    for a, b in (("<", "＜"), (">", "＞"), (":", "："), ('"', "”"),
                 ("/", "／"), ("\\", "＼"), ("|", "｜"), ("?", "？"), ("*", "＊")):
        name = name.replace(a, b)
    return name


def load_samples():
    return json.load(open(SAMPLE, encoding="utf-8"))


# ---------------- stage 1: download ----------------
def stage_download(limit):
    samples = load_samples()[:limit] if limit else load_samples()
    ok = skip = 0
    for s in samples:
        wid = s["id"]
        wdir = WORKDIR / f"RJ{wid}"
        if (wdir / "_download_done").exists():
            skip += 1
            continue
        wdir.mkdir(parents=True, exist_ok=True)
        try:
            tree = api_json(f"/tracks/{wid}?v=2")
            flat = walk_tracks(tree if isinstance(tree, list) else [])
            texts, audio = [], []
            for t, p, n in flat:
                url = n.get("mediaDownloadUrl")
                if not url:
                    continue
                if t == "text" and p.lower().endswith(".txt"):
                    texts.append((p, url, n.get("size", 0)))
                elif t == "audio":
                    audio.append((p, url, n.get("size", 0)))
            audio = dedup_audio(audio)
            (wdir / "script").mkdir(exist_ok=True)
            (wdir / "audio").mkdir(exist_ok=True)
            n_txt = n_aud = 0
            for p, url, size in texts:
                dest = wdir / "script" / safe_name(Path(p).name)
                if not dest.exists() or dest.stat().st_size == 0:
                    urllib.request.urlretrieve(url, dest)
                n_txt += 1
            for p, url, size in audio:
                dest = wdir / "audio" / safe_name(Path(p).name)
                if not dest.exists() or dest.stat().st_size == 0:
                    urllib.request.urlretrieve(url, dest)
                n_aud += 1
            (wdir / "_download_done").write_text(
                json.dumps({"texts": n_txt, "audio": n_aud}), encoding="utf-8")
            ok += 1
            print(f"[dl] RJ{wid}: txt={n_txt} audio={n_aud}", flush=True)
        except Exception as e:
            print(f"[dl][fail] RJ{wid}: {e}", flush=True)
    print(f"[download] 新下 {ok} 跳过 {skip}")


# ---------------- stage 2: asr ----------------
def stage_asr(limit):
    import livesub.config as lsc  # 复用生产管线的 CUDA DLL 注册（nvidia cublas/cudnn 轮子目录）
    lsc  # noqa: F821 — import 触发 _add_cuda_dlls()
    from faster_whisper import WhisperModel
    model = WhisperModel(str(MODEL_DIR), device="cuda", compute_type="float16")
    samples = load_samples()[:limit] if limit else load_samples()
    for s in samples:
        wdir = WORKDIR / f"RJ{s['id']}"
        adir = wdir / "audio"
        if not adir.exists():
            continue
        (wdir / "asr").mkdir(exist_ok=True)
        for mp3 in sorted(adir.glob("*.mp3")):
            out = wdir / "asr" / (mp3.stem + ".jsonl")
            if out.exists():
                continue
            segs, _ = model.transcribe(str(mp3), language="ja", beam_size=1, vad_filter=False)
            n = 0
            with open(out, "w", encoding="utf-8") as f:
                for seg in segs:
                    f.write(json.dumps({"start": round(seg.start, 2), "end": round(seg.end, 2),
                                        "text": seg.text.strip()}, ensure_ascii=False) + "\n")
                    n += 1
            print(f"[asr] RJ{s['id']}/{mp3.name}: {n} 段", flush=True)


# ---------------- stage 3: align ----------------
def script_sentences(txt_path):
    """台本 txt → 台词句列表（含假名汉字的行拆句）。
    舞台指示（整行括号拟音/场景说明）单独返回——它们没有对应语音，剔出覆盖率分母。"""
    sents, directions = [], 0
    for ln in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or not JA_KEEP.search(ln):
            continue
        # 整行舞台指示：全括号包裹（拟音/场景/心境）→ 无语音，不进分母
        stripped = re.sub(r"[（(].*?[)）]", "", ln).strip()
        if not stripped or not JA_KEEP.search(stripped):
            directions += 1
            continue
        if len(ln) < 3:
            continue
        for piece in SENT_SPLIT.split(ln):
            piece = piece.strip()
            if len(piece) >= 3 and JA_KEEP.search(piece):
                sents.append(piece)
    return sents


def stage_align(limit):
    samples = load_samples()[:limit] if limit else load_samples()
    for s in samples:
        wdir = WORKDIR / f"RJ{s['id']}"
        adir, sdir = wdir / "asr", wdir / "script"
        if not adir.exists() or not sdir.exists():
            continue
        (wdir / "align").mkdir(exist_ok=True)
        for txt in sorted(sdir.glob("*.txt")):
            out = wdir / "align" / (txt.stem + ".jsonl")
            if out.exists():
                continue
            # 找同前缀的 asr 文件（轨名对轨名，兜底=合并全部轨）
            asr_segs = []
            exact = adir / (txt.stem + ".jsonl")
            if exact.exists():
                src_files = [exact]
            else:
                src_files = sorted(adir.glob("*.jsonl"))
            for f in src_files:
                for line in open(f, encoding="utf-8"):
                    asr_segs.append(json.loads(line))
            if not asr_segs:
                continue
            sents = script_sentences(txt)
            results = []
            j = 0  # 单调推进指针：台本顺序=音频顺序，配对只许向前
            for si, sent in enumerate(sents):
                ns = norm_ja(sent)
                if not ns:
                    continue
                best, bi = 0.0, -1
                for k in range(j, min(j + 60, len(asr_segs))):  # 前向窗口 60 段
                    seg = asr_segs[k]
                    r = SequenceMatcher(None, ns, norm_ja(seg["text"])).ratio()
                    if r > best:
                        best, bi = r, k
                if best >= 0.35 and bi >= 0:
                    seg = asr_segs[bi]
                    results.append({"si": si, "sent": sent, "start": seg["start"],
                                    "end": seg["end"], "sim": round(best, 3)})
                    j = bi + 1
            with open(out, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cov = len(results) / max(len(sents), 1)
            print(f"[align] RJ{s['id']}/{txt.name}: 句 {len(sents)} 钉上 {len(results)} ({cov*100:.0f}%)", flush=True)


# ---------------- stage 4: report ----------------
def stage_report():
    rows = []
    for wdir in sorted(WORKDIR.glob("RJ*")):
        for f in (wdir / "align").glob("*.jsonl"):
            res = [json.loads(l) for l in open(f, encoding="utf-8")]
            for r in res:
                rows.append(r)
    if not rows:
        print("[report] 无对齐产物")
        return
    sims = sorted(r["sim"] for r in rows)
    n = len(sims)
    # 时间单调性：同文件内 start 倒退 = 钉歪
    mono_bad = 0
    for wdir in sorted(WORKDIR.glob("RJ*")):
        for f in (wdir / "align").glob("*.jsonl"):
            res = [json.loads(l) for l in open(f, encoding="utf-8")]
            mono_bad += sum(1 for a, b in zip(res, res[1:]) if b["start"] < a["start"] - 0.5)
    print(f"=== 台本对齐试产报告（聚合统计）===")
    print(f"钉上句数 {n} | 相似度均值 {sum(sims)/n:.2f} | 中位 {sims[n//2]:.2f} | p25 {sims[n//4]:.2f}")
    print(f"高置信(sim>=0.7) {sum(1 for x in sims if x>=0.7)*100//n}% | 中带(0.5~0.7) {sum(1 for x in sims if 0.5<=x<0.7)*100//n}% | 低带(0.35~0.5, 仅当文本语料) {sum(1 for x in sims if 0.35<=x<0.5)*100//n}%")
    print(f"时间单调违规(钉歪信号) {mono_bad} 处 ({mono_bad*100//n}%)")
    covs = []
    for wdir in sorted(WORKDIR.glob("RJ*")):
        w_cov = []
        for f in (wdir / "align").glob("*.jsonl"):
            res = [json.loads(l) for l in open(f, encoding="utf-8")]
            txt = wdir / "script" / (f.stem + ".txt")
            if txt.exists():
                total = len(script_sentences(txt))
                if total:
                    w_cov.append(len(res) / total)
        if w_cov:
            covs.append((wdir.name, sum(w_cov)/len(w_cov), len(w_cov)))
    print("\n每部覆盖率:")
    for name, cov, ntr in covs:
        print(f"  {name}: {cov*100:.0f}% ({ntr} 轨)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("download", "asr", "align", "report"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", default="", help="覆盖默认 40 部样本的清单 JSON 路径")
    args = ap.parse_args()
    global SAMPLE
    if args.sample:
        SAMPLE = Path(args.sample)
    WORKDIR.mkdir(exist_ok=True)
    {"download": lambda: stage_download(args.limit),
     "asr": lambda: stage_asr(args.limit),
     "align": lambda: stage_align(args.limit),
     "report": stage_report}[args.stage]()


if __name__ == "__main__":
    main()
