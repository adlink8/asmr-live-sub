import av
import wave
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR = 16000
DOWNLOADS = Path(r"D:\Downloads")

SOURCES = [
    list(DOWNLOADS.rglob("*02*こんな状況ですから*.mp3"))[0],
    list(DOWNLOADS.rglob("*03*原稿*.mp3"))[0],
    list(DOWNLOADS.rglob("*04*今日はずっとこの部屋で*.mp3"))[0],
]

def load_all_audio():
    print("=== 正在解码并拼合长音频音轨 ===")
    all_chunks = []
    for src in SOURCES:
        print(f"[*] 读取: {src.name}")
        sr = None
        with av.open(str(src)) as c:
            for frame in c.decode(audio=0):
                if sr is None:
                    sr = frame.sample_rate
                x = frame.to_ndarray().astype(np.float32)
                scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
                if x.ndim > 1:
                    x = x.mean(axis=0)
                all_chunks.append(x / scale)
    
    full = np.concatenate(all_chunks)
    if sr != TARGET_SR:
        print(f"[*] 统一重采样: {sr}Hz -> {TARGET_SR}Hz...")
        n_out = int(len(full) * TARGET_SR / sr)
        idx = np.linspace(0, len(full) - 1, n_out)
        full = np.interp(idx, np.arange(len(full)), full).astype(np.float32)
    return full

def save_clip(path: Path, data: np.ndarray, dur_s: float):
    n_samples = int(dur_s * TARGET_SR)
    clip = data[:n_samples]
    data16 = (np.clip(clip, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(data16.tobytes())
    dur = len(clip) / TARGET_SR
    print(f"    成功生成: {path.name} (时长: {dur:.1f}s / {dur/60:.1f}分钟, 大小: {path.stat().st_size / 1048576:.2f} MB)")

def main():
    audio = load_all_audio()
    total_dur = len(audio) / TARGET_SR
    print(f"总可用音频时长: {total_dur:.1f}s / {total_dur/60:.1f}分钟\n")

    print("[*] 正在构建 30 分钟音频 (1800s)...")
    save_clip(BENCH_DIR / "soak_30min.wav", audio, 1800.0)

    print("[*] 正在构建 1 小时音频 (3600s)...")
    save_clip(BENCH_DIR / "soak_60min.wav", audio, 3600.0)

if __name__ == "__main__":
    main()
