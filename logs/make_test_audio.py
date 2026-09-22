"""生成一段带真实停顿的日语测试音频，用于验证 sentence regrouping。

用 edge-tts 合成多句日语，句间插入**可控长度的停顿**：
  - 短停顿（0.5s）模拟"换气"，对应应被合并的半句
  - 长停顿（3.0s）模拟"讲完了"，对应应保持独立的句子

这样跑完整管线后，就能拿日志检查 regroup 规则是否只在短停顿处合并。

用法：python make_test_audio.py [输出路径]
"""
import asyncio
import io
import sys
import wave
from pathlib import Path

import edge_tts
import numpy as np

VOICE = "ja-JP-NanamiNeural"
SR = 24000

# (文本, 后面的停顿秒数)。0 表示句尾。
# 设计意图：
#  1) 「すごく気持ちいいですよ」+ 短停 + 「ね」 -> 应合并（终助词场景）
#  2) 「うん」+ 短停 + 长句 -> 应合并（相槌场景）
#  3) 长句 + 长停 + 长句 -> 不应合并
SCRIPT = [
    ("はい、聞こえていますよ", 0.5),
    ("今日も一日お疲れ様でした", 3.0),
    ("すごく気持ちいいですよ", 0.4),
    ("ね", 3.0),
    ("うん", 0.4),
    ("そうですね、私もそう思います", 3.0),
    ("ちょっと待ってください", 0.4),
    ("あ、ごめんなさい、忘れていました", 3.0),
    ("それでは続けましょう", 0.5),
    ("ゆっくりでいいですよ", 3.0),
    ("本当にありがとうございます", 3.0),
]


async def synth(text: str) -> np.ndarray:
    """合成为 float32 mono @ SR。"""
    comm = edge_tts.Communicate(text, VOICE)
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    # edge-tts 返回 MP3，用 av 解码
    import av
    with av.open(io.BytesIO(bytes(buf))) as c:
        parts = []
        rate = None
        for fr in c.decode(audio=0):
            if rate is None:
                rate = fr.sample_rate
            x = fr.to_ndarray().astype(np.float32)
            if x.ndim > 1:
                x = x.mean(axis=0)
            parts.append(x)
    if not parts:
        return np.zeros(0, np.float32)
    x = np.concatenate(parts)
    if rate != SR:
        n = int(len(x) * SR / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    # 归一化到 ~0.3 峰值，贴近正常语音能量
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak * 0.3).astype(np.float32)


async def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "test_convo.wav")
    rng = np.random.default_rng(0)
    parts = []
    # 开头 1s 静音，让 Segmenter 的 gate 先热身
    parts.append(np.full(SR, 1e-4, np.float32))
    for text, gap in SCRIPT:
        x = await synth(text)
        parts.append(x)
        print(f"  + {text!r}  audio={len(x)/SR:.2f}s  停顿={gap}s")
        if gap > 0:
            # 静音加一点极低底噪，模拟真实环境
            n = int(SR * gap)
            parts.append((rng.standard_normal(n) * 1e-4).astype(np.float32))
    parts.append(np.full(SR, 1e-4, np.float32))
    full = np.concatenate(parts)

    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(full, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"\n[out] {out}  总长 {len(full)/SR:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
