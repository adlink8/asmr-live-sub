import av
import numpy as np
from pathlib import Path

DOWNLOADS = Path(r"D:\Downloads")
TARGET_SR = 16000
FRAME_S = 0.51

def analyze_track_density(mp3_path: Path):
    chunks = []
    sr = None
    try:
        with av.open(str(mp3_path)) as c:
            for frame in c.decode(audio=0):
                if sr is None:
                    sr = frame.sample_rate
                x = frame.to_ndarray().astype(np.float32)
                scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
                if x.ndim > 1:
                    x = x.mean(axis=0)
                chunks.append(x / scale)
                # 检查前 300 秒 (5分钟)
                if len(chunks) * len(x) / sr > 300.0:
                    break
    except Exception as e:
        return None

    if not chunks:
        return None
        
    full = np.concatenate(chunks)
    frame_len = int(sr * FRAME_S)
    n_frames = len(full) // frame_len
    if n_frames == 0:
        return None

    voiced_flags = []
    for i in range(n_frames):
        frame = full[i*frame_len:(i+1)*frame_len]
        rms = float(np.sqrt(np.mean(frame * frame)))
        voiced_flags.append(rms > 0.01) # 语音能量门限

    density = sum(voiced_flags) / len(voiced_flags)
    
    # 统计停顿长度分布（用于判断是连续连说还是随机零星说）
    runs = []
    curr_silence = 0
    for v in voiced_flags:
        if not v:
            curr_silence += 1
        else:
            if curr_silence > 0:
                runs.append(curr_silence * FRAME_S)
                curr_silence = 0
    if curr_silence > 0:
        runs.append(curr_silence * FRAME_S)

    avg_gap = np.mean(runs) if runs else 0.0
    max_gap = np.max(runs) if runs else 0.0
    return {
        "name": mp3_path.name,
        "path": mp3_path,
        "density": round(density * 100, 1),
        "avg_gap_s": round(avg_gap, 2),
        "max_gap_s": round(max_gap, 2),
        "total_dur_s": round(len(full) / sr, 1)
    }

def main():
    print("=== 扫描本地音频人声密度与停顿特征 ===")
    mp3s = list(DOWNLOADS.rglob("*.mp3"))
    valid = [p for p in mp3s if ("Blue Archive" in str(p) or "hololive" in str(p)) and p.stat().st_size > 10*1048576]
    
    results = []
    for p in valid:
        r = analyze_track_density(p)
        if r:
            results.append(r)

    results.sort(key=lambda x: x["density"], reverse=True)
    print(f"\n{'音轨名称':<35} | {'人声占比(密度)':<14} | {'平均停顿间隔':<12} | {'最大静音'}")
    print("-" * 75)
    for r in results:
        print(f"{r['name'][:32]:<35} | {r['density']:<5}%        | {r['avg_gap_s']:<5}s      | {r['max_gap_s']}s")

if __name__ == "__main__":
    main()
