import av
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000

SOURCES = [
    {
        "name": "scene_a_dialogue.wav",
        "desc": "Scene A: 连续耳语念白（测试长句延迟与谓语完整度）",
        "file": r"D:\Downloads\Blue Archive ASMR (Vol 8)\RJ01547273 - Iroha\mp3\03.『暇なら遊びに来ません？』.mp3",
        "start_s": 30.0,
        "duration_s": 90.0,
    },
    {
        "name": "scene_b_interactive.wav",
        "desc": "Scene B: 间歇互动相槌（测试短句防吞字）",
        "file": r"D:\Downloads\Blue Archive ASMR (Vol 8)\RJ01547273 - Iroha\mp3\02.『お仕事大変そーですね？』.mp3",
        "start_s": 45.0,
        "duration_s": 90.0,
    },
    {
        "name": "scene_c_whisper_breath.wav",
        "desc": "Scene C: 弱音耳语与气声（测试抗幻觉与防死循环）",
        "file": r"D:\Downloads\Blue Archive ASMR (Vol 8)\RJ01547273 - Iroha\mp3\04.『こんな時くらいは』.mp3",
        "start_s": 60.0,
        "duration_s": 90.0,
    },
]

def extract_clip(src_path: str, start_s: float, dur_s: float) -> np.ndarray:
    chunks = []
    sr = None
    with av.open(src_path) as c:
        for frame in c.decode(audio=0):
            if sr is None:
                sr = frame.sample_rate
            x = frame.to_ndarray().astype(np.float32)
            scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
            if x.ndim > 1:
                x = x.mean(axis=0)
            chunks.append(x / scale)
    full = np.concatenate(chunks)
    
    # Resample to 16000
    if sr != TARGET_SR:
        n_out = int(len(full) * TARGET_SR / sr)
        idx = np.linspace(0, len(full) - 1, n_out)
        full = np.interp(idx, np.arange(len(full)), full).astype(np.float32)
    
    start_idx = int(start_s * TARGET_SR)
    end_idx = int((start_s + dur_s) * TARGET_SR)
    clip = full[start_idx:end_idx]
    return clip

def write_wav(path: Path, data: np.ndarray, rate=TARGET_SR):
    import wave
    data16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data16.tobytes())

def main():
    print("=== 生成 ASMR 黄金标准测试套件 (Benchmark Suite) ===")
    for spec in SOURCES:
        out_file = BENCH_DIR / spec["name"]
        print(f"[*] 提取: {spec['desc']} -> {spec['name']}")
        clip = extract_clip(spec["file"], spec["start_s"], spec["duration_s"])
        write_wav(out_file, clip)
        print(f"    完成: 时长={len(clip)/TARGET_SR:.1f}s 采样率={TARGET_SR}Hz 文件大小={out_file.stat().st_size} bytes")
    print("\n所有 3 类场景的测试素材已全部构建完毕！")

if __name__ == "__main__":
    main()
