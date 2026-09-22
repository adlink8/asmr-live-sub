import av
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR = 16000

def find_file(parent: Path, pattern: str) -> Path:
    matches = list(parent.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} in {parent}")
    return matches[0]

DOWNLOADS = Path(r"D:\Downloads")

SPECS = [
    {
        "id": "scene_a_dialogue",
        "desc": "Scene A: 连续长句念白（蔚蓝档案 Vol 10 剧情台词轨，测延迟与完整度）",
        "pattern": "*02*こんな状況ですから*.mp3",
        "start_s": 60.0,
        "duration_s": 180.0, # 3 分钟
    },
    {
        "id": "scene_b_interactive",
        "desc": "Scene B: 间歇互动相槌（Hololive 耳かきおもてなし轨，测短相槌防吞字）",
        "pattern": "*耳かきで癒しのおもてなし*.mp3",
        "start_s": 60.0,
        "duration_s": 180.0, # 3 分钟
    },
    {
        "id": "scene_c_whisper_soft",
        "desc": "Scene C: 弱音耳语与伴睡（蔚蓝档案 Vol 10 弱音轨，测抗幻觉与防死循环）",
        "pattern": "*04*今日はずっとこの部屋で*.mp3",
        "start_s": 90.0,
        "duration_s": 180.0, # 3 分钟
    }
]

def extract_clip(src_path: Path, start_s: float, dur_s: float) -> np.ndarray:
    chunks = []
    sr = None
    with av.open(str(src_path)) as c:
        for frame in c.decode(audio=0):
            if sr is None:
                sr = frame.sample_rate
            x = frame.to_ndarray().astype(np.float32)
            scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
            if x.ndim > 1:
                x = x.mean(axis=0)
            chunks.append(x / scale)
            # Early stop decoding when we have enough data (dur_s + start_s + 10s buffer)
            if len(chunks) * len(x) / sr > (start_s + dur_s + 10.0):
                break
    full = np.concatenate(chunks)
    
    if sr != TARGET_SR:
        n_out = int(len(full) * TARGET_SR / sr)
        idx = np.linspace(0, len(full) - 1, n_out)
        full = np.interp(idx, np.arange(len(full)), full).astype(np.float32)
    
    s_idx = int(start_s * TARGET_SR)
    e_idx = int((start_s + dur_s) * TARGET_SR)
    return full[s_idx:e_idx]

def write_wav(path: Path, data: np.ndarray, rate=TARGET_SR):
    import wave
    data16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data16.tobytes())

def main():
    print("=== 构建 3 场景 ASMR 黄金标准考卷 (每段 180 秒) ===")
    for spec in SPECS:
        src = find_file(DOWNLOADS, spec["pattern"])
        out_wav = BENCH_DIR / f"{spec['id']}.wav"
        print(f"[*] 提取: {spec['desc']}")
        print(f"    源文件: {src.name}")
        clip = extract_clip(src, spec["start_s"], spec["duration_s"])
        write_wav(out_wav, clip)
        dur = len(clip) / TARGET_SR
        print(f"    成功 -> {out_wav.name} (时长: {dur:.1f}s, 大小: {out_wav.stat().st_size / 1048576:.2f} MB)\n")
    print("全部 3 个测试音频切片构建完毕！")

if __name__ == "__main__":
    main()
