"""LLM 语义裁判：judge_input.json → semantic_judge.json（双口径：语义分 + 拦截数据）。

配方（沿用 2026-09-22/23 step-5-preview 裁判口径，.workbuddy/memory 记录）：
- meaning-only：判"意思是否正确"，不抠措辞与译风；
- 判分三档：correct / partial / wrong；歧义以日文原文（JA）仲裁；
- 拟声词/纯计数类 cue 标 kind=sound/countdown，整类移出语义分母
  （连判对的一起剔——只剔判错的是按结果挑数据），单列为拦截数据；
- 裁判会错且 n 小：数字读作估计与退步报警，不读作精确真值。

两种后端（二选一）：
1. --api    OpenAI 兼容 chat completions。环境变量：
              SEMANTIC_JUDGE_BASE_URL（如 https://api.z.ai/v1）
              SEMANTIC_JUDGE_API_KEY
              SEMANTIC_JUDGE_MODEL（如 step-5-preview）
            批量送对，要求返回 JSON 数组。若 key 缺失/调用失败 → 退出码 2，
            场景停在 judge_input.json（严禁伪造裁判结果）。
2. --verdicts <file>  裁判 verdict 清单（会话内 LLM 按上述口径逐对判好后落盘的
            JSON 数组：seg_id/verdict/kind/note）。本模式只做校验 + 汇总 +
            写 schema 标准的 semantic_judge.json——判分本体仍来自真实 LLM
            判断，脚本不生成任何 verdict。

kind 预判（启发式，仅作 API 模式提示与 verdicts 模式缺省，判分者可覆盖）：
  countdown=纯计数；sound=拟声/纯语气；其余 dialogue。

用法：
  python tools/run_semantic_judge.py scene_asmr299717_t02 --api
  python tools/run_semantic_judge.py scene_asmr299717_t02 --verdicts v.json
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

INTERCEPTION_NOTE = ("拟声/纯计数 cue 整类移出语义分母（连正确的一起剔）；"
                     "单列为拦截数据=过滤器/ASR 非言语区漏放质量")

COUNT_RE = re.compile(r"^[\d〇零一二两三四五六七八九十百千万半]+[つ个個只枚度回歳才]*$")
SOUND_RE = re.compile(r"^[\s～~—…・。．、！!？?\-—♪♫ぁぃぅぇぉっゃゅょァィゥェォッャュョ"
                      r"啊嗚呜嗯呣唔哩呐呢啦呀哇哦噢喔哎欸诶嘿呵哈呼哼えおあいうんのねよな"
                      r"はぁ-ぉんヴ]+$")
SENT_END = re.compile(r"[。！？!?…〜~]$")


def guess_kind(human_zh: str) -> str:
    t = re.sub(r"\s+", "", human_zh or "")
    if not t:
        return "sound"
    if COUNT_RE.fullmatch(t):
        return "countdown"
    if len(t) <= 12 and SOUND_RE.fullmatch(t):
        return "sound"
    return "dialogue"


def build_rubric(pairs):
    lines = [
        "你是日译中实时字幕的语义裁判。对每一对给出 verdict。",
        "输入字段：seg_id、ja（实时 ASR 的日文，可能有误）、sakura_live（实时译文）、human（人工中文参考=意图锚点）。",
        "判分口径（meaning-only）：",
        "- correct：译文传达了参考句的核心意思（不抠措辞/语气/译风；同义改写算对）。",
        "- partial：只传达了部分意思，或有明显偏差（漏了半句、指代错位、语气错但大意在）。",
        "- wrong：意思错误、答非所问，或与参考句完全无关。",
        "- 若 ja 与 human 都看不懂对齐关系，以 ja 原文仲裁：译文忠实于 ja 而与 human 不符时，",
        "  在 note 里注明\"参考疑错位\"并判 correct（译文没错）。",
        "每对另标 kind：dialogue=对白/叙述（进语义分母）；sound=拟声/纯语气；countdown=纯计数。",
        "sound/countdown 的对也要判 verdict（用于拦截数据统计），但会移出语义分母。",
        "note：一两句话说明理由，正确且无疑的可留空。",
        "只输出 JSON 数组：[{\"seg_id\":..,\"verdict\":\"correct|partial|wrong\",\"kind\":\"dialogue|sound|countdown\",\"note\":\"...\"}]",
        "待判数据：",
        json.dumps(pairs, ensure_ascii=False),
    ]
    return "\n".join(lines)


def judge_via_api(pairs, base_url, api_key, model, batch=10):
    verdicts = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": build_rubric(chunk)}],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                m = re.search(r"\[.*\]", text, re.S)
                verdicts.extend(json.loads(m.group(0)))
                break
            except (urllib.error.URLError, KeyError, json.JSONDecodeError,
                    urllib.error.HTTPError) as e:
                if attempt == 2:
                    raise RuntimeError(f"judge API 调用持续失败: {e}") from e
                time.sleep(5)
    return verdicts


def validate(verdicts, pairs):
    """校验 verdict 覆盖与取值合法；kind 缺省时用启发式预判。"""
    by_id = {p["seg_id"]: p for p in pairs}
    out = {}
    for v in verdicts:
        sid = v.get("seg_id")
        if sid not in by_id:
            raise ValueError(f"verdict 引用了不存在的 seg_id {sid}")
        verdict = str(v.get("verdict") or v.get("sakura") or "").lower()
        if verdict not in ("correct", "partial", "wrong"):
            raise ValueError(f"seg_id {sid} verdict 非法: {verdict!r}")
        kind = v.get("kind") or guess_kind(by_id[sid]["human"])
        if kind not in ("dialogue", "sound", "countdown"):
            kind = "dialogue"
        out[sid] = {"seg_id": sid, "sakura": verdict, "kind": kind,
                    "note": v.get("note") or ""}
    missing = [p["seg_id"] for p in pairs if p["seg_id"] not in out]
    if missing:
        raise ValueError(f"verdict 缺少 seg_id: {missing}")
    return [out[p["seg_id"]] for p in pairs]


def summarize(verdicts):
    def bucket(vs):
        b = {"correct": 0, "partial": 0, "wrong": 0}
        for v in vs:
            b[v["sakura"]] += 1
        n = len(vs)
        b["n"] = n
        b["correctPct"] = (round(b["correct"] / n * 100, 1) if n else None)
        return b

    dialogue = [v for v in verdicts if v["kind"] == "dialogue"]
    interception = {}
    for kind in ("sound", "countdown"):
        vs = [v for v in verdicts if v["kind"] == kind]
        if vs:
            b = bucket(vs)
            interception[kind] = {k: b[k] for k in ("n", "correct", "partial", "wrong")}
    return bucket(verdicts), bucket(dialogue), interception


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--judge-input", default=None,
                    help="默认 dataset/<scene>.judge_input.json")
    ap.add_argument("--out", default=None,
                    help="默认 dataset/<scene>.semantic_judge.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--api", action="store_true",
                   help="用 SEMANTIC_JUDGE_* 环境变量指定的 OpenAI 兼容 API 判分")
    g.add_argument("--verdicts", help="会话内 LLM 判好的 verdict 清单 JSON")
    args = ap.parse_args()

    ji_p = Path(args.judge_input) if args.judge_input else DATASET / f"{args.scene}.judge_input.json"
    pairs = json.loads(ji_p.read_text(encoding="utf-8"))["pairs"]
    if not pairs:
        sys.exit(f"[NG] {ji_p} 无配对")

    if args.api:
        base = os.environ.get("SEMANTIC_JUDGE_BASE_URL", "")
        key = os.environ.get("SEMANTIC_JUDGE_API_KEY", "")
        model = os.environ.get("SEMANTIC_JUDGE_MODEL", "")
        if not (base and key and model):
            print("[NG] 缺 SEMANTIC_JUDGE_BASE_URL / API_KEY / MODEL——"
                  "不伪造裁判结果，场景停在 judge_input.json", file=sys.stderr)
            sys.exit(2)
        method = (f"LLM semantic judge ({model} via API), meaning-only, "
                  "JA as ambiguity arbiter")
        verdicts = validate(judge_via_api(pairs, base, key, model), pairs)
    else:
        raw = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
        verdicts = validate(raw, pairs)
        method = ("LLM semantic judge (in-session LLM following step-5-preview "
                  "recipe, verdicts via --verdicts), meaning-only, "
                  "JA as ambiguity arbiter")

    summary, summary_dialogue, interception = summarize(verdicts)
    out = {
        "scene": args.scene,
        "method": method,
        "built_at": time.strftime("%Y-%m-%d"),
        "total_pairs": len(pairs),
        "summary": {"sakura_live": summary},
        "summary_dialogue": {"sakura_live": summary_dialogue},
        "interception": interception,
        "interception_note": INTERCEPTION_NOTE,
        "verdicts": verdicts,
    }
    out_p = Path(args.out) if args.out else DATASET / f"{args.scene}.semantic_judge.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {out_p.name}: n={summary['n']} 语义 n={summary_dialogue['n']} "
          f"correct={summary_dialogue['correctPct']}% 拦截={interception}")


if __name__ == "__main__":
    main()
