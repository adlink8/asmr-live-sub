import pyaudiowpatch as pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()
loops = list(p.get_loopback_device_info_generator())

print("正在通过非阻塞 Callback 探测各个输出设备当前声音信号 (2秒)...")

results = {}
streams = []

def make_cb(name):
    def cb(in_data, frame_count, time_info, status):
        x = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        results[name]["frames"].append(x)
        return (None, pyaudio.paContinue)
    return cb

for d in loops:
    idx = d["index"]
    name = d["name"]
    rate = int(d["defaultSampleRate"])
    ch = max(1, int(d["maxInputChannels"]))
    
    if any(k in name for k in ["Steam", "网易", "ToDesk"]):
        continue
    
    results[name] = {"frames": [], "idx": idx, "rate": rate}
    try:
        s = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                   input_device_index=idx, frames_per_buffer=1024,
                   stream_callback=make_cb(name))
        s.start_stream()
        streams.append(s)
    except Exception as e:
        print(f"无法打开设备 [{idx}] {name}: {e}")

time.sleep(1.5)

for s in streams:
    try:
        s.stop_stream()
        s.close()
    except Exception:
        pass

p.terminate()

print("\n--- 探测结果汇总 ---")
for name, data in results.items():
    if not data["frames"]:
        print(f"[{data['idx']}] {name}: 没有收到任何音频流回调 (静默/未工作)")
        continue
    arr = np.concatenate(data["frames"])
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(arr**2)))
    status = "★★【正在播放声音！】★★" if peak > 0.005 else "静音/无明显声音"
    print(f"[{data['idx']}] {name}: peak={peak:.4f}, rms={rms:.4f} -> {status}")
