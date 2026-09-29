"""ASMR 声学仿真与加噪混音工具

基于 numpy 和 scipy 实现零外部依赖的 ASMR 声学环境仿真：
把纯净的 TTS 耳语人声，与真实的 ASMR 掏耳/舔舐水声音效底轨进行动态信噪比 (SNR) 混合，
生成兼具 100% 准确台词真值与极限抗噪特性的训练切片。

使用方式：
    .venv\\Scripts\\python.exe tools/simulate_asmr_audio.py --clean "oneesan_comfort.wav" --noise "D:/Downloads/asmr-sub-corpus" --snr 5.0 --out "simulated_asmr.wav"
"""
import argparse
import os
import random
from pathlib import Path
import numpy as np
from scipy.io import wavfile


def load_audio(path: str) -> tuple[int, np.ndarray]:
    """读取 WAV 文件并统一归一化为 float32 [-1.0, 1.0]"""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    return sr, data


def calculate_rms(samples: np.ndarray) -> float:
    """计算音频均方根能量"""
    return float(np.sqrt(np.mean(samples ** 2) + 1e-12))


def mix_audio_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按指定信噪比 (SNR dB) 将人声与背景噪声混合
    SNR = 10 * log10(P_signal / P_noise)
    """
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if noise_rms == 0 or clean_rms == 0:
        return clean

    # 目标噪声能量
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms

    scaled_noise = noise * scale
    mixed = clean + scaled_noise

    # 软峰值防爆音 (Soft clipping / Limiter)
    max_val = np.max(np.abs(mixed))
    if max_val > 0.98:
        mixed = mixed / max_val * 0.98

    return mixed


def simulate(clean_path: str, noise_source: str, snr_db: float, out_path: str):
    clean_sr, clean_data = load_audio(clean_path)
    clean_len = len(clean_data)

    # 如果是立体声转单声道做基底
    if clean_data.ndim > 1:
        clean_data = clean_data.mean(axis=1)

    # 寻找背景噪声文件
    noise_path = Path(noise_source)
    if noise_path.is_dir():
        wav_files = list(noise_path.rglob("*.wav"))
        if not wav_files:
            print(f"[warn] 目录 {noise_source} 下未找到 .wav 文件，降级使用白噪音仿真")
            noise_data = np.random.normal(0, 0.05, clean_len).astype(np.float32)
        else:
            selected_noise_file = random.choice(wav_files)
            print(f"[sim] 随机抽样背景音效底轨: {selected_noise_file.name}")
            noise_sr, noise_raw = load_audio(str(selected_noise_file))
            if noise_raw.ndim > 1:
                noise_raw = noise_raw.mean(axis=1)
            # 截取或循环至相同长度
            if len(noise_raw) < clean_len:
                repeats = (clean_len // len(noise_raw)) + 1
                noise_data = np.tile(noise_raw, repeats)[:clean_len]
            else:
                start = random.randint(0, len(noise_raw) - clean_len)
                noise_data = noise_raw[start:start + clean_len]
    else:
        noise_sr, noise_raw = load_audio(str(noise_path))
        if noise_raw.ndim > 1:
            noise_raw = noise_raw.mean(axis=1)
        if len(noise_raw) < clean_len:
            repeats = (clean_len // len(noise_raw)) + 1
            noise_data = np.tile(noise_raw, repeats)[:clean_len]
        else:
            noise_data = noise_raw[:clean_len]

    # 混合
    print(f"[sim] 注入声学噪声 (信噪比 SNR: {snr_db:.1f} dB)...")
    mixed = mix_audio_with_snr(clean_data, noise_data, snr_db)

    # 保存为 16-bit PCM WAV
    out_int16 = (mixed * 32767.0).astype(np.int16)
    wavfile.write(out_path, clean_sr, out_int16)
    print(f"[sim] 仿真合成完成！输出已落盘: {out_path} (时长: {clean_len/clean_sr:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASMR 声学仿真混音器")
    parser.add_argument("--clean", default="oneesan_comfort.wav", help="TTS 生成的干净耳语人声路径")
    parser.add_argument("--noise", default="dataset", help="背景音效来源 (单文件或目录)")
    parser.add_argument("--snr", type=float, default=5.0, help="信噪比 (dB)，越小噪声越剧烈，负数代表水声压制人声")
    parser.add_argument("--out", default="simulated_asmr.wav", help="输出混音路径")
    args = parser.parse_args()

    simulate(args.clean, args.noise, args.snr, args.out)
