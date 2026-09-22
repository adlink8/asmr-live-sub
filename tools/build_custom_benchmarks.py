import av
import numpy as np
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR = 16000
DOWNLOADS = Path(r"D:\Downloads")

def find_file(parent: Path, pattern: str) -> Path:
    matches = list(parent.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} in {parent}")
    return matches[0]

CUSTOM_SPECS = [
    {
        "id": "scene_dense_talk",
        "desc": "Scene Dense: 高频密集对话（Hololive 特典闲聊杂谈，3人快速轮流说话，极高人声密度 99%）",
        "pattern": "*特典_アフタートークボイス*.mp3",
        "start_s": 5.0,
        "duration_s": 180.0,
    },
    {
        "id": "scene_random_talk",
        "desc": "Scene Random: 随机节奏说话（蔚蓝档案 圣园未夜轨，忽快忽慢、随机停顿掏耳与插科打诨，人声密度 66%）",
        "pattern": "*05.『水は器に従ひて』*.mp3",
        "start_s": 20.0,
        "duration_s": 180.0,
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
    data16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data16.tobytes())

def main():
    print("=== 构建【高频说话】与【随机说话】测试音频 (每段 180 秒) ===")
    for spec in CUSTOM_SPECS:
        src = find_file(DOWNLOADS, spec["pattern"])
        out_wav = BENCH_DIR / f"{spec['id']}.wav"
        print(f"[*] 提取: {spec['desc']}")
        print(f"    源文件: {src.name}")
        clip = extract_clip(src, spec["start_s"], spec["duration_s"])
        write_wav(out_wav, clip)
        dur = len(clip) / TARGET_SR
        print(f"    成功 -> {out_wav.name} (时长: {dur:.1f}s, 大小: {out_wav.stat().st_size / 1048576:.2f} MB)\n")
    print("测试音频提取构建完毕！")

if __name__ == "__main__":
    main()
