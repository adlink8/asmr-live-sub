import argparse
import json
import re
import sys
from pathlib import Path
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def clean_script_line(line: str) -> str:
    # 移除角色名标记如 【未夜】、[Mika]、(耳かき) 等剧本指示
    line = re.sub(r"^[【\[\(（].*?[】\]\)）]\s*", "", line)
    # 移除行首行尾空白和音效说明
    line = re.sub(r"（.*?）|\(.*?\)", "", line)
    return line.strip()

def align_script_to_audio(script_path: Path, live_jsonl_path: Path, out_json: Path):
    print(f"\n=======================================================")
    print(f"[*] 启动官方剧本台本对齐器 (Scheme B: Official Script Alignment)")
    print(f"    官方剧本文件: {script_path.name}")
    print(f"    输入实时流式日志: {live_jsonl_path.name}")
    
    with open(script_path, "r", encoding="utf-8", errors="replace") as f:
        script_lines = [clean_script_line(l) for l in f if clean_script_line(l)]
        
    events = []
    with open(live_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: events.append(json.loads(line))
                except: pass
                
    asr_evs = [e for e in events if e.get("kind") == "asr" and e.get("ja")]

    aligned_segments = []
    curr_script_idx = 0

    for asr in asr_evs:
        sid = asr.get("seg_id")
        ja = asr.get("ja", "").strip()
        t0_s = asr.get("t_start") or 0.0
        t1_s = asr.get("t_end") or 0.0
        
        # 拟声词处理（如 じゅるるる、ズズズズ）
        is_sound_effect = bool(re.search(r"^[じズずッ・…\s]+$", ja))
        
        best_match = None
        best_score = 0
        best_offset = 0
        
        if not is_sound_effect and curr_script_idx < len(script_lines):
            window = script_lines[curr_script_idx: min(len(script_lines), curr_script_idx + 4)]
            for offset, cand in enumerate(window):
                score = fuzz.partial_ratio(ja, cand)
                if score > best_score:
                    best_score = score
                    best_match = cand
                    best_offset = offset
                    
        if best_score >= 50 and best_match:
            chosen_official = best_match
            # 如果匹配很完整，前移剧本游标
            if fuzz.ratio(ja, best_match) >= 70 or best_score >= 85:
                curr_script_idx += best_offset + 1
        elif is_sound_effect:
            chosen_official = ja # 拟声词以真实环境音为主
        else:
            chosen_official = ja
            
        # 不可评段：拟声词（剧本不标注，拿识别结果当真值是自我兑现）与
        # 低置信匹配（<50，对齐不可信）。这些段不进 CER，由 evaluate 按
        # seg_id 两侧同时剔除。
        if is_sound_effect:
            excluded, reason = True, "sound_effect"
        elif best_score < 50:
            excluded, reason = True, "low_confidence_match"
        else:
            excluded, reason = False, ""

        aligned_segments.append({
            "idx": len(aligned_segments) + 1,
            "seg_id": sid,
            "t0": t0_s,
            "t1": t1_s,
            "official_ja": chosen_official,
            "live_ja": ja,
            "match_score": round(best_score, 1),
            "excluded": excluded,
            "exclude_reason": reason,
        })
        
    result = {
        "script_file": script_path.name,
        "audio_file": live_jsonl_path.name,
        "total_segments": len(aligned_segments),
        "full_official_ja": "".join(s["official_ja"] for s in aligned_segments),
        "segments": aligned_segments
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[✓] 官方剧本基准成功对齐并持久化至: {out_json}")
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="Path to official script txt")
    ap.add_argument("--live", required=True, help="Path to live jsonl")
    ap.add_argument("--out", required=True, help="Output json path")
    args = ap.parse_args()
    
    align_script_to_audio(Path(args.script), Path(args.live), Path(args.out))

if __name__ == "__main__":
    main()
