"""生成 dataset/manifest.json——数据集的标准索引。

扫描 dataset/*.gt_ja.json 与 *.ref_zh.json，核对配对完整性与音频存在性，
把每个 scene 的 tier、覆盖率、provenance、可用指标、已知缺陷汇总成机器可读索引。
评测消费方（evaluate_accuracy.py 与人）以本文件判断"哪个 scene 能测什么"。

层级定义：
  gold    官方台本对齐（外部权威，唯一可作准确度声明的真值）
  teacher 离线全篇 beam=5 解码（与实时管线共享模型偏差；时间戳不可靠；
          跨运行非确定——钉死具体文件，不重生成；且漏检无界：scene_c
          实测漏掉 3 处真实台词，live 多识别的真实内容全被计成 Ins，
          数字指标对它不成立，只配人读 diff 表与跑延迟/效率基准）
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "dataset"

AUDIO_SOURCE = {
    "scene_a": "Blue Archive Vol 8 (RJ01547273 - Iroha) mp3/03『暇なら遊びに来ません？』",
    "scene_b": "Blue Archive Vol 8 (RJ01547273 - Iroha) mp3/02『お仕事大変そーですね？』",
    "scene_c": "Blue Archive Vol 8 (RJ01547273 - Iroha) mp3/04『こんな時くらいは』",
    "scene_dense_talk": "hololive「天使の止まり木」特典_アフタートークボイス（3 人杂谈）",
    "scene_random_talk": "Blue Archive 圣园未夜轨 05.『水は器に従ひて』",
}

AUDIO_FILE = {
    "scene_a": "scene_a_dialogue.wav",
    "scene_b": "scene_b_interactive.wav",
    "scene_c": "scene_c_whisper_soft.wav",
    "scene_dense_talk": "scene_dense_talk.wav",
    "scene_random_talk": "scene_random_talk.wav",
}

LIVE_RUNS = {
    "scene_a": ["baseline_scene_a", "adaptive_5s_scene_a", "hardcap_5s_scene_a"],
    "scene_b": ["baseline_scene_b", "adaptive_5s_scene_b", "hardcap_5s_scene_b"],
    "scene_c": ["baseline_scene_c", "adaptive_5s_scene_c", "hardcap_5s_scene_c"],
    "scene_dense_talk": ["custom_dense_talk"],
    "scene_random_talk": ["custom_random_talk"],
}

# 版权边界：DLsite/同人作品音轨不推送远端仓库
COPYRIGHT_NOTE = ("音轨来自 DLsite 同人作品（Blue Archive / hololive 同人 ASMR），"
                  "含授权边界：本目录只存真值/参考译文等元数据，音频留 benchmarks/，"
                  "任何情况下不推远端公开仓库。")


def scene_entry(name):
    gt_p = DATASET / f"{name}.gt_ja.json"
    if not gt_p.exists():
        raise FileNotFoundError(f"缺少 {gt_p.name}，数据集不完整")
    gt = json.loads(gt_p.read_text(encoding="utf-8"))
    segs = gt.get("segments", [])
    kept = [s for s in segs if not s.get("excluded")]
    excl = {}
    for s in segs:
        if s.get("excluded"):
            excl[s["exclude_reason"]] = excl.get(s["exclude_reason"], 0) + 1

    tier = "gold" if gt.get("script_file") else "teacher"
    audio = ROOT / "benchmarks" / AUDIO_FILE[name]
    if not audio.exists():
        raise FileNotFoundError(f"缺少音频 {audio}")

    caveats = []
    if tier == "teacher":
        caveats.append("teacher 级：与实时管线共享模型偏差")
        caveats.append("时间戳不可靠（t1≈t0+0.9s 与内容无关），逐句对照表仅供参考")
        caveats.append("跨运行非确定（CT2 GPU 浮点归并），本文件是钉死的快照，"
                       "重新生成会得到不同的退化形态")
        caveats.append("**漏检无界**：全篇 beam=5 无 VAD 解码会整段漏掉真实语音"
                       "（scene_c 实测漏 3 处真实台词），live 多识别出的真实内容"
                       "全被计成 Ins——CER 97.52% 是参考漏检的产物，不是管线错误")
        caveats.append("非言语区（吟唱/气声）被离线解码转成文字，而实时管线"
                       "在这些区输出拟声词，两侧不同构")
    else:
        caveats.append("金级：官方台本为外部权威，可作准确度声明")
        caveats.append("台本为手工整理，覆盖不全——低置信匹配段已 excluded")

    # 参考层可有多份：ref_zh（Sakura 精译，确定性）、ref_zh_stepflash
    # （云端模型，采样不可复现）、human_zh（人工译文，唯一绝对锚点）。
    # 按 dataset/<name>.ref*.json / <name>.human_zh.json glob 发现。
    refs = []
    for rp in sorted(DATASET.glob(f"{name}.ref*.json")):
        try:
            rd = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        prov = rd.get("provenance", {})
        model = str(prov.get("model", ""))
        refs.append({
            "file": f"dataset/{rp.name}",
            "kind": ("cloud_reference" if "step" in model or "gpt" in model.lower()
                     else "quality_reference"),
            "provenance": prov,
            "deterministic": bool(prov.get("deterministic", True)),
        })
    human_p = DATASET / f"{name}.human_zh.json"
    human_zh = (f"dataset/{name}.human_zh.json"
                if human_p.exists() else None)

    return {
        "name": name,
        "audio": {
            "file": f"benchmarks/{AUDIO_FILE[name]}",
            "duration_s": 180.0,
            "source": AUDIO_SOURCE[name],
        },
        "gt": {
            "file": f"dataset/{name}.gt_ja.json",
            "tier": tier,
            "method": ("official_script_alignment" if tier == "gold"
                       else "offline_fullpass_beam5_no_vad"),
            "segments_total": len(segs),
            "segments_kept": len(kept),
            "excluded": excl,
            "gt_ja_chars": sum(len(s.get("ja") or s.get("official_ja") or "")
                               for s in kept),
        },
        "refs": refs,
        "human_zh": human_zh,
        "live_runs": [f"logs/benchmark_runs/{r}.jsonl" for r in LIVE_RUNS[name]],
        # teacher 级不产数字指标：参考漏检无界（live 多识别的真实内容会被
        # 计成 Ins），CER/MT 相似度对它是无意义的测量，只配人读 diff 表
        "valid_for": (["asr_cer", "mt_similarity"] if tier == "gold"
                      else ["qualitative_diff", "latency_efficiency"]),
        "caveats": caveats,
    }


def mt_experiment_entry(judge_p):
    """人工译文锚点轨道（asmr.one 路线）：无 JA 真值，只做 MT 端到端对标。

    与 scenes 不同轨，不能混用：JA 输入是实时 ASR 输出（非真值），参考是
    人工中文字幕，度量是 LLM 语义裁判的意思正确率——回答"端到端字幕贴近
    人工意图多少"，不回答 ASR 准不准。数字全部从 semantic_judge.json 实文件
    读入，不手抄。
    """
    j = json.loads(judge_p.read_text(encoding="utf-8"))
    name = judge_p.name.replace(".semantic_judge.json", "")
    files = {
        "human_zh": f"dataset/{name}.human_zh.json",
        "ja_input": f"dataset/{name}.live_ja.json",
        "judge_input": f"dataset/{name}.judge_input.json",
    }
    if (ROOT / "dataset" / f"{name}.ref_zh_stepflash.json").exists():
        files["cloud_ref"] = f"dataset/{name}.ref_zh_stepflash.json"
    return {
        "name": name,
        "kind": "human_translation_anchor",
        "method": j.get("method"),
        "total_pairs": j.get("total_pairs"),
        "summary": j.get("summary"),
        "summary_dialogue": j.get("summary_dialogue"),
        "interception": j.get("interception"),
        "files": files,
        "live_run": f"logs/benchmark_runs/{name}.jsonl",
        "audio": {"in_repo": False,
                  "note": "asmr.one 商业作品音轨，版权内容只留本地语料库"
                          "（D:/Downloads/asmr-zh-corpus/），任何情况不入库不推远端"},
        "ja_input_caveat": "JA 输入是实时 ASR 输出，非真值——本轨道只测"
                           "端到端翻译意思正确率，不测 ASR 准确度",
    }


def main():
    names = sorted(p.name.replace(".gt_ja.json", "")
                   for p in DATASET.glob("*.gt_ja.json"))
    scenes = [scene_entry(n) for n in names]
    experiments = [mt_experiment_entry(p)
                   for p in sorted(DATASET.glob("*.semantic_judge.json"))]

    neg_p = ROOT / "benchmarks" / "negative_samples.json"
    negatives = {"file": "benchmarks/negative_samples.json",
                 "count": len(json.loads(neg_p.read_text(encoding="utf-8"))
                              .get("samples", [])) if neg_p.exists() else 0,
                 "purpose": "防幻觉对照组：全段应零字幕"}

    manifest = {
        "dataset": "asmr-live-sub accuracy benchmark",
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema": {
            "scene": "一个 180s 评测场景 = audio + gt_ja（真值日文）+ ref_zh（离线精译参考）",
            "gt_tiers": {
                "gold": "官方台本对齐，外部权威，唯一可作准确度声明",
                "teacher": "离线全篇 beam=5 解码，共享模型偏差，只配测管线损失",
            },
            "ref": "gt_ja 的 Sakura 贪心精译（temp=0/max_tokens=512/rp=1.12），"
                   "与具体 live 运行解耦，确定性输出",
            "excluded": "真值段级剔除标记，evaluate_accuracy 双侧按 seg_id 同时剔除",
            "mt_experiments": "人工译文锚点轨道（无 JA 真值）：JA 输入是实时 ASR "
                              "输出，参考是 asmr.one 人工中文字幕，度量是 LLM "
                              "语义裁判的意思正确率——只测端到端翻译，不测 ASR",
        },
        "scenes": scenes,
        "mt_experiments": experiments,
        "negatives": negatives,
        "copyright": COPYRIGHT_NOTE,
        "consumption": (
            "evaluate_accuracy.py --live <jsonl> --gt dataset/<scene>.gt_ja.json "
            "--mt-ref dataset/<scene>.ref_zh.json --out <report.md>；"
            "**只有 gold 级场景的数字可作准确度声明**，teacher 级的 CER/MT "
            "相似度不成立（参考漏检无界，实测 scene_c 97.52% 全是漏检产物），"
            "只配人读逐句 diff 表 + 跑延迟/效率基准"),
    }
    out = DATASET / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[OK] manifest: {out}  ({len(scenes)} scenes)")
    for s in scenes:
        g = s["gt"]
        print(f"  {s['name']:20s} tier={g['tier']:7s} "
              f"kept={g['segments_kept']}/{g['segments_total']} "
              f"ja_chars={g['gt_ja_chars']} valid={','.join(s['valid_for'])}")
    for e in experiments:
        sm = e.get("summary") or {}
        print(f"  {e['name']:20s} kind={e['kind']} pairs={e['total_pairs']} "
              f"summary={json.dumps(sm, ensure_ascii=False)[:80]}")
    print(f"  negatives: {negatives['count']}")


if __name__ == "__main__":
    main()
