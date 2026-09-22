import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    out_lines = []
    def dump_md(p, title):
        p = Path(p)
        if not p.exists(): return
        with open(p, encoding='utf-8') as f:
            evs = [json.loads(l) for l in f if l.strip()]
        asr_map = {e.get('seg_id'): e for e in evs if e.get('kind') == 'asr'}
        mts = [e for e in evs if e.get('kind') == 'mt' and e.get('zh')]
        out_lines.append(f"\n### 【{title}】 (共 {len(mts)} 条字幕)")
        out_lines.append("| # | 时间轴位置 | 段长 | 端到端总延迟 | ASR/MT 耗时 | 日文原文 (ASR) | 中文译文 (Sakura-7B) |")
        out_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for i, m in enumerate(mts, 1):
            sid = m.get('seg_id')
            asr = asr_map.get(sid, {})
            t0 = asr.get('t_start') or 0.0
            t1 = asr.get('t_end') or 0.0
            d = asr.get('audio_s') or 0.0
            asr_s = m.get('asr_s') or 0.0
            mt_s = m.get('mt_s') or 0.0
            tot = round(d + asr_s + mt_s, 2)
            ja = m.get('ja', '').replace('|', '\\|')
            zh = m.get('zh', '').replace('|', '\\|')
            out_lines.append(f"| {i} | {t0:.1f}s ~ {t1:.1f}s | {d:.1f}s | **{tot:.2f}s** | asr:{asr_s:.2f}s mt:{mt_s:.2f}s | `{ja}` | **{zh}** |")

    dump_md(ROOT / 'logs' / 'soak_runs' / 'adaptive_15m.jsonl', '15 分钟长时压测流 (蔚蓝档案 剧情轨)')
    dump_md(ROOT / 'logs' / 'marathon_runs' / '30m_adaptive.jsonl', '30 分钟连续马拉松 (蔚蓝档案 剧情轨)')
    dump_md(ROOT / 'logs' / 'benchmark_runs' / 'adaptive_5s_scene_a.jsonl', 'Scene A 连续剧情对话 (蔚蓝档案 台词轨)')
    dump_md(ROOT / 'logs' / 'benchmark_runs' / 'adaptive_5s_scene_b.jsonl', 'Scene B 间歇互动相槌 (Hololive 天使耳语轨)')
    dump_md(ROOT / 'logs' / 'benchmark_runs' / 'adaptive_5s_scene_c.jsonl', 'Scene C 弱音耳语伴睡 (蔚蓝档案 弱音轨)')

    with open(ROOT / 'logs' / 'subtitles_table.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')

if __name__ == '__main__':
    main()
