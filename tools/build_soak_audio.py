import av
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR = 16000

DOWNLOADS = Path(r"D:\Downloads")
matches = list(DOWNLOADS.rglob("*03*原稿*.mp3"))
if not matches:
    matches = list(DOWNLOADS.rglob("*耳かきで癒しのおもてなし*.mp3"))
SRC_MP3 = matches[0]
OUT_WAV = BENCH_DIR / "soak_15min.wav"

START_S = 60.0
DUR_S = 900.0  # 15 分钟

def main():
    print(f"=== 构建 15 分钟长时压测素材 (Soak Audio) ===")
    print(f"源文件: {SRC_MP3.name}")
    print(f"提取区间: {START_S}s ~ {START_S + DUR_S}s (共 15 分钟)")
    
    chunks = []
    sr = None
    with av.open(str(SRC_MP3)) as c:
        for frame in c.decode(audio=0):
            if sr is None:
                sr = frame.sample_rate
            x = frame.to_ndarray().astype(np.float32)
            scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
            if x.ndim > 1:
                x = x.mean(axis=0)
            chunks.append(x / scale)
            if len(chunks) * len(x) / sr > (START_S + DUR_S + 15.0):
                break
                
    full = np.concatenate(chunks)
    if sr != TARGET_SR:
        n_out = int(len(full) * TARGET_SR / sr)
        idx = np.linspace(0, len(full) - 1, n_out)
        full = np.interp(idx, np.arange(len(full)), full).astype(np.float32)
    
    s_idx = int(START_S * TARGET_SR)
    e_idx = int((START_S + DUR_S) * TARGET_SR)
    clip = full[s_idx:e_idx]
    
    import wave
    data16 = (np.clip(clip, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(OUT_WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(data16.tobytes())
        
    dur = len(clip) / TARGET_SR
    print(f"成功构建: {OUT_WAV.name} (时长: {dur:.1f}s / {dur/60:.1f}分钟, 大小: {OUT_WAV.stat().st_size / 1048576:.2f} MB)")

if __name__ == "__main__":
    main()
