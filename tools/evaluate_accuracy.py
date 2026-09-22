import argparse
import json
import re
import sys
from pathlib import Path
import jiwer
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def normalize_ja(text: str) -> str:
    if not text: return ""
    # 去除多余标点与空白，仅保留纯字符流用于严格 CER 比较
    text = re.sub(r"[\s\t\r\n、。！？!?…~―・]+", "", text)
    return text.strip()

def normalize_zh(text: str) -> str:
    if not text: return ""
    text = re.sub(r"[\s\t\r\n，。！？!?…~—·]+", "", text)
    return text.strip()

def evaluate(live_jsonl: Path, gt_json: Path, out_md: Path, mt_ref_json: Path = None):
    print(f"\n=======================================================")
    print(f"[*] 启动 ASR 与 MT 准确度自动化评测 (Automated Evaluation Engine)")
    print(f"    实时流式日志: {live_jsonl.name}")
    print(f"    黄金真值基准: {gt_json.name}")
    print(f"    MT 参考译文: {mt_ref_json.name if mt_ref_json else '未提供（MT 指标将不可评）'}\n")

    with open(gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # MT 参考译文必须来自独立来源（离线全篇翻译）。
    # 历史上真值文件的 zh 直接抄自同一次实时运行，拿它比 live 是自比自，
    # 恒为 100%，那个指标已废除。
    ref_zh_segments = []
    if mt_ref_json:
        with open(mt_ref_json, "r", encoding="utf-8") as f:
            ref_zh_segments = json.load(f).get("segments", [])

    # 读取 live_jsonl
    events = []
    with open(live_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: events.append(json.loads(line))
                except: pass

    asr_evs = [e for e in events if e.get("kind") == "asr" and e.get("ja")]
    mt_evs = [e for e in events if e.get("kind") == "mt" and e.get("zh") and not e.get("zh_refusal")]
    mt_map = {e.get("seg_id"): e for e in mt_evs}

    live_segments = []
    for asr in asr_evs:
        sid = asr.get("seg_id")
        mt = mt_map.get(sid, {})
        live_segments.append({
            "seg_id": sid,
            "t0": asr.get("t_start") or 0.0,
            "t1": asr.get("t_end") or 0.0,
            "audio_s": asr.get("audio_s") or 0.0,
            "ja": asr.get("ja", "").strip(),
            "zh": mt.get("zh", "").strip(),
            "asr_s": asr.get("asr_s") or 0.0,
            "mt_s": mt.get("mt_s") or 0.0
        })

    gt_segments = gt_data.get("segments", [])

    # excluded 段不参评（align 标记的拟声词/低置信匹配、curate 标记的解码乱码）：
    # 真值侧直接剔除；实时侧按时间重叠找最匹配的真值段，落在 excluded 段上的
    # live 段也剔——否则 live 在拟声区的真实输出会被计成 Ins，且 excluded 段
    # 的 official_ja 抄的是 live 自己的文本，留着只会稀释分母。
    def best_overlap_gt(l_t0, l_t1):
        best_ov, best_g = 0.0, None
        for g in gt_segments:
            ov = max(0.0, min(l_t1, g.get("t1", 0.0)) - max(l_t0, g.get("t0", 0.0)))
            if ov > best_ov:
                best_ov, best_g = ov, g
        return best_ov, best_g

    gt_kept = [g for g in gt_segments if not g.get("excluded")]
    live_eval = []
    n_live_dropped = 0
    for ls in live_segments:
        ov, g = best_overlap_gt(ls["t0"], ls["t1"])
        if g is not None and ov > 0.3 and g.get("excluded"):
            n_live_dropped += 1
            continue
        live_eval.append(ls)

    # 1. 全文级宏观 CER（仅未剔除段）
    gt_full_ja = normalize_ja("".join(s.get("official_ja") or s.get("ja") or "" for s in gt_kept))
    live_full_ja = normalize_ja("".join(s["ja"] for s in live_eval))

    cer_result = jiwer.process_characters(gt_full_ja, live_full_ja)
    total_chars = len(gt_full_ja)
    cer_pct = round(cer_result.cer * 100, 2)
    sub_pct = round(cer_result.substitutions / total_chars * 100, 2) if total_chars else 0.0
    del_pct = round(cer_result.deletions / total_chars * 100, 2) if total_chars else 0.0
    ins_pct = round(cer_result.insertions / total_chars * 100, 2) if total_chars else 0.0
    acc_pct = round(max(0.0, (1.0 - cer_result.cer) * 100), 2)

    # 2. 全文级宏观中文翻译相似度：live 全文 vs 独立参考译文全文
    live_full_zh = normalize_zh("".join(s["zh"] for s in live_eval))
    if ref_zh_segments:
        ref_full_zh = normalize_zh("".join(s.get("zh") or "" for s in ref_zh_segments))
        zh_similarity = round(fuzz.ratio(ref_full_zh, live_full_zh), 2)
    else:
        zh_similarity = None

    print(f"\n【评测核心指标】")
    n_gt_excl = len(gt_segments) - len(gt_kept)
    print(f"  * 真值参考总字数: {total_chars} 字符"
          f"（真值剔除 {n_gt_excl} 段，live 侧同步剔除 {n_live_dropped} 段，不参评）")
    print(f"  * ASR 字符正确率 (Accuracy) : {acc_pct}%")
    print(f"  * ASR 字符错误率 (CER)      : {cer_pct}%")
    print(f"      - 替换错字 (Substitutions) : {cer_result.substitutions} 字 ({sub_pct}%)")
    print(f"      - 吞字漏字 (Deletions)     : {cer_result.deletions} 字 ({del_pct}%)")
    print(f"      - 幻觉多字 (Insertions)    : {cer_result.insertions} 字 ({ins_pct}%)")
    if zh_similarity is None:
        print("  * MT 相似度: 不可评（未提供 --mt-ref 独立参考译文）\n")
    else:
        print(f"  * MT 相似度 (实时 vs {mt_ref_json.name}): {zh_similarity}%\n")

    # 3. 逐段对齐与差分生成
    diff_rows = []
    for live in live_segments:
        l_t0, l_t1 = live["t0"], live["t1"]
        l_ja = live["ja"]
        l_zh = live["zh"]
        
        # 寻找在时间窗上重叠度最大的 GT 段
        matched_gt = []
        for g in gt_segments:
            gt_t0, gt_t1 = g.get("t0", 0.0), g.get("t1", 0.0)
            # 判重叠
            overlap = max(0.0, min(l_t1, gt_t1) - max(l_t0, gt_t0))
            if overlap > 0.3 or (gt_t0 >= l_t0 - 0.5 and gt_t1 <= l_t1 + 0.5):
                matched_gt.append(g)
                
        if not matched_gt:
            # fallback by text ratio
            scored = sorted([(fuzz.partial_ratio(l_ja, g.get("official_ja") or g.get("ja")), g) for g in gt_segments], key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > 60:
                matched_gt = [scored[0][1]]
                
        gt_ja_text = "".join(g.get("official_ja") or g.get("ja") or "" for g in matched_gt)
        # 匹配到的真值段全部被剔除 → 本行不参评，标记原因供人查
        row_excluded = bool(matched_gt) and all(g.get("excluded") for g in matched_gt)
        if row_excluded:
            gt_ja_text = f"（剔除：{matched_gt[0].get('exclude_reason', 'excluded')}）"
        # 参考中文只认独立参考译文（--mt-ref），按时间重叠匹配，匹配不到留空
        gt_zh_text = ""
        if ref_zh_segments and not row_excluded:
            best_ov, best_zh = 0.0, ""
            for g in ref_zh_segments:
                ov = max(0.0, min(l_t1, g.get("t1", 0.0)) - max(l_t0, g.get("t0", 0.0)))
                if ov > best_ov:
                    best_ov, best_zh = ov, (g.get("zh") or "").strip()
            if best_ov > 0.3:
                gt_zh_text = best_zh

        # 单句 CER（剔除行不算）
        seg_cer = (round(jiwer.cer(normalize_ja(gt_ja_text), normalize_ja(l_ja)) * 100, 1)
                   if gt_ja_text and not row_excluded else None)
        
        diff_rows.append({
            "t0": l_t0,
            "t1": l_t1,
            "live_ja": l_ja,
            "gt_ja": gt_ja_text,
            "cer": seg_cer,
            "live_zh": l_zh,
            "gt_zh": gt_zh_text,
        })

    # 生成 Markdown 报告
    md_lines = []
    md_lines.append(f"# ASR 识别与 MT 翻译准确度量化对比报告\n")
    md_lines.append(f"- **评测对象**: `{live_jsonl.name}`")
    md_lines.append(f"- **黄金真值**: `{gt_json.name}`\n")
    md_lines.append(f"## 一、核心准确度量化总览\n")
    md_lines.append(f"| 指标项 | 测量值 | 工业判定标准 | 说明 |")
    md_lines.append(f"| :--- | :--- | :--- | :--- |")
    md_lines.append(f"| **ASR 字符准确率** | **{acc_pct}%** | ≥90% 优秀 | 整体字符命中率 |")
    md_lines.append(f"| **ASR 字符错误率 (CER)** | **{cer_pct}%** | ≤10% 生产可用 | 标准 `jiwer` 算得整体字符错率 |")
    md_lines.append(f"| └ 替换错字率 (Sub) | {sub_pct}% ({cer_result.substitutions}字) | - | 同音异形词、近音字替换 |")
    md_lines.append(f"| └ **吞字漏识别率 (Del)** | **{del_pct}%** ({cer_result.deletions}字) | **≤5% 安全线** | **最关键指标**：切块是否斩断词尾/吞掉促音 |")
    md_lines.append(f"| └ 幻觉多字率 (Ins) | {ins_pct}% ({cer_result.insertions}字) | ≤3% 安全线 | VAD 或切块引起的无中生有 |")
    if zh_similarity is None:
        md_lines.append("| **实时译文 vs 离线参考译文相似度** | **不可评** | ≥75% 优秀 | 未提供 `--mt-ref`：真值侧无独立参考译文，自比自指标已废除 |\n")
    else:
        md_lines.append(f"| **实时译文 vs 离线参考译文相似度** | **{zh_similarity}%** | ≥75% 优秀 | 实时切块翻译与 `{mt_ref_json.name}` 全篇离线译文的一致性 |\n")

    # 逐行参考列只有在参考译文能按时间对齐时才渲染。
    # gt_teacher_*.json 的时间戳不可靠（存在 0.12s 内一大段文本的段），
    # 对不上就整列省略，MT 差异以上方全文指标为准，不留空列误导。
    has_ref = any(r["gt_zh"] for r in diff_rows)
    md_lines.append(f"## 二、逐句真值差分对照表 (Subtitles Diff Detail)\n")
    if has_ref:
        md_lines.append(f"| 时间窗 | 实时日文 (Live ASR) | 真值日文 (Ground Truth) | 句级CER | 实时中文 (Live MT) | 离线参考译文 (--mt-ref) |")
        md_lines.append(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
    else:
        md_lines.append(f"| 时间窗 | 实时日文 (Live ASR) | 真值日文 (Ground Truth) | 句级CER | 实时中文 (Live MT) |")
        md_lines.append(f"| :--- | :--- | :--- | :--- | :--- |")
        if ref_zh_segments:
            md_lines.append(f"\n> 逐行参考译文未能按时间对齐（参考译文时间戳不可靠），MT 差异以上方全文指标为准。\n")
    for r in diff_rows:
        l_ja = r["live_ja"].replace("|", "\\|")
        g_ja = r["gt_ja"].replace("|", "\\|")
        l_zh = r["live_zh"].replace("|", "\\|")
        g_zh = r["gt_zh"].replace("|", "\\|") if r["gt_zh"] else "—"
        cer_v = r["cer"]
        cer_badge = ("**{}%**".format(cer_v) if cer_v is not None and cer_v > 15
                     else ("{}%".format(cer_v) if cer_v is not None else "n/a"))
        if has_ref:
            md_lines.append(f"| {r['t0']:.1f}s~{r['t1']:.1f}s | {l_ja} | {g_ja} | {cer_badge} | {l_zh} | {g_zh} |")
        else:
            md_lines.append(f"| {r['t0']:.1f}s~{r['t1']:.1f}s | {l_ja} | {g_ja} | {cer_badge} | {l_zh} |")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[✓] 准确度量化报告已生成至: {out_md}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True, help="Live sub jsonl path")
    ap.add_argument("--gt", required=True, help="Ground truth json path (JA)")
    ap.add_argument("--mt-ref", default=None, help="Independent reference translation json (offline full-pass); MT metric is unrated without it")
    ap.add_argument("--out", required=True, help="Output markdown path")
    args = ap.parse_args()

    evaluate(Path(args.live), Path(args.gt), Path(args.out),
             Path(args.mt_ref) if args.mt_ref else None)

if __name__ == "__main__":
    main()
