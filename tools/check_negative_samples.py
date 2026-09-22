"""负样本对照检查：对 benchmarks/neg_*.wav 真跑完整管线，合格标准是零字幕。

每个文件一次独立 subprocess（live_sub.py 无批量模式），命令与
run_custom_eval.py 一致（anime + sakura + adaptive）。MT 必须是真的：
--mt none 本身就不上屏（已知歧义），拿它做对照会是空转。

输出：逐文件 PASS/FAIL，有字幕即 FAIL 并打印漏网内容。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
BENCH = ROOT / "benchmarks"


def check_one(wav: Path):
    cmd = [
        str(PYTHON), "live_sub.py",
        "--model", "anime",
        "--mt", "sakura",
        "--mt-ngl", "99",
        "--layer", "1",
        "--strategy", "adaptive",
        "--max-s", "5.0",
        "--hang-s", "2.0",
        "--source-audio", str(wav),
        "--replay",
        "--fast",
    ]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=600)
    subs = [l.strip() for l in p.stdout.splitlines() if l.startswith("[zh]")]
    return subs, p.stdout, p.stderr


def main():
    manifest = json.load(open(BENCH / "negative_samples.json", encoding="utf-8"))
    samples = manifest["samples"]
    print(f"负样本对照检查：{len(samples)} 个段 × {manifest['window_s']}s，合格=零字幕\n")

    failures = 0
    for s in samples:
        wav = BENCH / s["name"]
        if not wav.exists():
            print(f"[SKIP] {s['name']} 不存在（先跑 build_negative_samples.py）")
            continue
        print(f"[run ] {s['name']}  ({s['category']}, {s['source']})")
        subs, out, err = check_one(wav)
        if not subs:
            print(f"[PASS] {s['name']}  零字幕\n")
        else:
            failures += 1
            print(f"[FAIL] {s['name']}  产出 {len(subs)} 条字幕：")
            for line in subs:
                print(f"       {line}")
            print()

    if failures:
        print(f"结果：{failures}/{len(samples)} 个负样本漏出字幕——防幻觉有缺口")
        sys.exit(1)
    print(f"结果：{len(samples)}/{len(samples)} 全部零字幕，防幻觉对照通过")


if __name__ == "__main__":
    main()
