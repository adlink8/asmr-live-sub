"""Gemini 3.8 Flash TTS 音频生成工具

调用 Google Gemini 3.8 Flash TTS API 生成逼真的近场贴耳 ASMR 耳语干音。
使用方式：
    $env:GEMINI_KEY="你的API_KEY"
    .venv\\Scripts\\python.exe tools/gemini_tts_generate.py --text "台词" --out "output.wav"
"""
import argparse
import base64
import json
import os
import sys
import urllib.request


def generate_tts(text: str, out_path: str, api_key: str = None):
    key = api_key or os.environ.get("GEMINI_KEY")
    if not key:
        print("[error] 未检测到 GEMINI_KEY，请传入参数或设置环境变量 $env:GEMINI_KEY", file=sys.stderr)
        sys.exit(1)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash-tts:generateContent?key={key}"

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"]
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print(f"[tts] 正在请求 Gemini 3.8 Flash TTS 生成音频 (文本长度: {len(text)} 字)...")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            cand = res["candidates"][0]
            part = cand["content"]["parts"][0]
            if "inlineData" in part:
                audio_bytes = base64.b64decode(part["inlineData"]["data"])
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"[tts] 生成成功！文件保存至: {out_path} ({len(audio_bytes)} 字节)")
                return out_path
            else:
                print(f"[error] 未获取到音频数据: {cand}", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        if hasattr(e, "read"):
            print(f"[error] API 报错: {e.read().decode('utf-8')}", file=sys.stderr)
        else:
            print(f"[error] 请求失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini 3.8 Flash TTS 生成器")
    parser.add_argument("--text", default="[whispers] …お疲れ様。[short pause] 今日も、本当によく頑張ったね。[slow] 私の前では、無理しなくていいんだよ…[sighs] ほら、こっちおいで…", help="输入的台词文本")
    parser.add_argument("--out", default="oneesan_comfort.wav", help="输出音频路径 (.wav)")
    parser.add_argument("--key", default=None, help="Gemini API Key（默认读环境变量 GEMINI_KEY）")
    args = parser.parse_args()

    generate_tts(args.text, args.out, args.key)
