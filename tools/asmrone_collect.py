"""asmr.one 带字幕作品采集器（数据集金级素材来源）。

API 地面真值来自 D:\\ADLINK\\Myproject\\seanime 的 internal/api/asmr（2026-09-13 实测）：
  搜索 GET /api/search/{page}?subtitle=1&pageSize=（无需鉴权；keyword 参数服务端忽略）
  音轨 GET /api/tracks/{id}?v=2 → 树形节点 type ∈ {folder, audio, text, image}
  媒体 mediaDownloadUrl（raw.kiko-play-niptan.one 直链，无需鉴权）

价值：asmr.one 上大量作品带**人工中文字幕（LRC/VTT，带时间戳）**——正是数据集缺的
human_zh 锚点（绝对翻译质量第一次可测）；部分作品同时带日文字幕，则 JA 金级真值
也一并解决。中文字幕的文件夹命名混乱（LyRiCs字幕数据 / 大家一起来翻译_台本 /
中国語簡体字 / みんなで翻訳均在列），**文件夹名不可信，本工具按字幕内容检测语言**。

用法：
  # 扫描（下载样本文本判语言，不落盘素材）
  python tools/asmrone_collect.py scan --pages 10 --out D:/Downloads/asmr-zh-corpus

  # 下载入选作品（字幕必下，音频可选）
  python tools/asmrone_collect.py fetch --from-inv D:/Downloads/asmr-zh-corpus/inventory.json ^
      --limit 5 --audio --max-audio-mb 300 --out D:/Downloads/asmr-zh-corpus
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.asmr.one/api"
UA = "li/adlink8 (seanime-personal-fork)"
THROTTLE = 0.35  # 与 seanime client 的 300ms 限流礼仪一致

KANA_RE = re.compile("[぀-ヿ]")
HAN_RE = re.compile("[一-鿿]")

# 成人内容标题过滤词（用户要"正常 asmr"；字幕结构上与成人作品无异，按标题粗筛）
ADULT_RE = re.compile(
    "オナサポ|搾精|射精|精液|手コキ|フェラ|アナル|セックス|痴女|凌辱|"
    "寝取|人妻|調教|奴隷|変態|淫|R18|18禁|痴漢|強制|"
    "サキュバス|催眠|洗脳|触手|妊娠|孕ませ|中出|近親相姦|肉棒|自慰")


def api_get(path, binary=False, retries=2, headers=None):
    url = path if path.startswith("http") else API + path
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = r.read()
            return data if binary else data.decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"GET {url} 失败: {last}")


def api_json(path, headers=None):
    return json.loads(api_get(path, headers=headers))


def walk_tracks(nodes, path=""):
    """-> [(type, full_path, node)]"""
    out = []
    for n in nodes:
        t = n.get("type", "")
        p = path + "/" + (n.get("title") or "")
        out.append((t, p, n))
        if n.get("children"):
            out.extend(walk_tracks(n["children"], p))
    return out


def detect_lang(text):
    """按假名/汉字比例判语言：中文字幕常含片假名人名（パトラ等），只要有假名就算
    JA 会把中文全错判过去——日语句子假名占比高（>25%），中文行是汉字主导。"""
    kana = len(KANA_RE.findall(text))
    han = len(HAN_RE.findall(text))
    if han == 0 and kana == 0:
        return "other"
    if han == 0:
        return "ja"
    if kana / (kana + han) > 0.25:
        return "ja"
    return "zh"


def parse_sub_text(raw):
    """LRC/VTT/TXT -> 纯文本行（去时间戳与头部），用于语言检测。"""
    out = []
    for ln in raw.splitlines():
        ln = ln.strip().lstrip("﻿")
        if not ln or ln.startswith(("[ti:", "[re:", "[ve:", "[ar:", "[al:",
                                  "WEBVTT", "NOTE", "Kind:", "Language:")):
            continue
        ln = re.sub(r"\\[\\d{2}:\\d{2}[.,]\\d{2,3}\\]", "", ln)           # LRC 时间戳
        ln = re.sub(r"^\\d{2}:\\d{2}:\\d{2}[.,]\\d{3}\\s*-->.*$", "", ln)  # VTT 时间行
        if ln and not ln.isdigit():
            out.append(ln)
    return "\n".join(out)


def dedup_audio(audio):
    """全音轨去重：同一轨常同时挂 mp3 和 wav 两套，按文件名 stem 分组，
    mp3 优先（体积小 5~10 倍，ASR 输入反正重采样 16k），其余格式只补 mp3 没有的轨。
    audio: [(path, url, size)] -> [(path, url, size)]"""
    groups = {}
    for p, url, size in audio:
        stem = Path(p).stem.lower()
        groups.setdefault(stem, []).append((p, url, size))
    out = []
    for stem, items in groups.items():
        mp3 = [x for x in items if x[0].lower().endswith(".mp3")]
        pick = mp3[0] if mp3 else items[0]
        out.append(pick)
    return out


def classify_work(wid, title):
    """拉音轨树，逐文本文件下载判语言。无中文字幕 -> None。"""
    tree = api_json(f"/tracks/{wid}?v=2")
    flat = walk_tracks(tree if isinstance(tree, list) else [])
    zh_files, ja_files, audio = [], [], []
    for t, p, n in flat:
        url = n.get("mediaDownloadUrl")
        if not url:
            continue
        if t == "text":
            try:
                body = parse_sub_text(api_get(url, binary=True)
                                      .decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                continue
            lang = detect_lang(body)
            if lang == "zh":
                zh_files.append((p, url))
            elif lang == "ja":
                ja_files.append((p, url))
        elif t == "audio":
            audio.append((p, url, n.get("size") or 0))
    if not zh_files:
        return None
    return {
        "id": wid, "title": title,
        "zh_files": [p for p, _ in zh_files],
        "zh_urls": [u for _, u in zh_files],
        "ja_sub_count": len(ja_files),
        "ja_urls": [u for _, u in ja_files],
        "audio_count": len(audio),
        "audio_bytes": sum(s for _, _, s in audio),
        # 去重后（mp3/wav 二选一）的实际下载量，夜间筛选用这个
        "audio_bytes_unique": sum(s for _, _, s in dedup_audio(audio)),
    }


def cmd_scan(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    found, scanned, seen = [], 0, set()
    for page in range(1, args.pages + 1):
        d = api_json(f"/search/{page}?subtitle=1&pageSize={args.page_size}"
                     f"&order={args.order}")
        works = d.get("works", [])
        if not works:
            break
        for w in works:
            wid = w["id"]
            if wid in seen:
                continue
            seen.add(wid)
            scanned += 1
            title = w.get("title", "")
            if args.normal and ADULT_RE.search(title):
                continue
            try:
                info = classify_work(wid, title)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {wid} 失败: {e}")
                time.sleep(THROTTLE)
                continue
            if info:
                info.update({
                    "nsfw": bool(w.get("nsfw")),
                    "release": w.get("release", ""),
                    "rating": w.get("rate_average_2dp", 0),
                    "dl_count": w.get("dl_count", 0),
                    "duration_min": w.get("duration", 0),
                    # 官方 tag 全集（[{id,name}] -> 名字列表），标签补全打分的原料
                    "tags": [t.get("name") for t in (w.get("tags") or []) if t.get("name")],
                })
                found.append(info)
                print(f"  [hit] {wid} zh×{len(info['zh_urls'])} "
                      f"ja×{info['ja_sub_count']} audio×{info['audio_count']} "
                      f"{title[:38]}")
            time.sleep(THROTTLE)
    inv = out_dir / "inventory.json"
    inv.write_text(json.dumps(
        {"scanned": scanned, "with_zh_subtitle": len(found),
         "with_ja_sub": sum(1 for f in found if f["ja_sub_count"]),
         "works": found}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ja = json.loads(inv.read_text(encoding="utf-8"))["with_ja_sub"]
    print(f"\n[OK] 扫描 {scanned} 部（去重后），带中文字幕 {len(found)} 部，"
          f"其中同日文字幕 {n_ja} 部")
    print(f"     清单: {inv}")


def cmd_fetch(args):
    inv = json.loads(Path(args.from_inv).read_text(encoding="utf-8"))
    works = [w for w in inv["works"] if w.get("zh_urls")]
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        works = [w for w in works if w["id"] in want]
    # 优先：有日文字幕的（JA 金级+ZH 人工双全），再按下载量
    works.sort(key=lambda w: (w["ja_sub_count"] > 0, w["dl_count"]), reverse=True)
    works = works[: args.limit]
    out_dir = Path(args.out)
    for w in works:
        wdir = out_dir / f"RJ{w['id']}"
        subs = wdir / "subtitles"
        subs.mkdir(parents=True, exist_ok=True)
        meta = {"id": w["id"], "title": w["title"],
                "release": w.get("release", ""), "nsfw": w.get("nsfw"),
                "rating": w.get("rating", 0),
                "tags": w.get("tags", [])}
        n_dl = 0
        seen_urls = set()
        for url in w["zh_urls"] + w.get("ja_urls", []):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1]) or "sub.txt"
            sub_dir = subs / ("ja" if url in w.get("ja_urls", []) else "zh")
            sub_dir.mkdir(exist_ok=True)
            try:
                data = api_get(url, binary=True)
                (sub_dir / name).write_bytes(data)
                n_dl += 1
                print(f"  [dl] {'ja' if sub_dir.name == 'ja' else 'zh'}/{name} "
                      f"({len(data)}B)")
                time.sleep(THROTTLE)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {name} 下载失败: {e}")
        if args.audio_full:
            # 全音轨：去重后（mp3 优先）逐条下载，单条超限的跳过
            try:
                tree = api_json(f"/tracks/{w['id']}?v=2")
                auds = [(p, n.get("mediaDownloadUrl"), n.get("size") or 0)
                        for t, p, n in walk_tracks(tree if isinstance(tree, list) else [])
                        if t == "audio" and n.get("mediaDownloadUrl")]
                auds = dedup_audio(auds)
                cap = args.max_audio_mb * 1024 * 1024
                adir = wdir / "audio"
                adir.mkdir(exist_ok=True)
                n_audio = 0
                for _, url, size in auds:
                    if size > cap:
                        print(f"  [skip] {url.rsplit('/', 1)[-1]} "
                              f"({size // 1024 // 1024}MB > 单条上限)")
                        continue
                    name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
                    data = api_get(url, binary=True)
                    (adir / name).write_bytes(data)
                    n_audio += 1
                    print(f"  [dl] audio/{name} ({len(data) // 1024}KB)")
                    time.sleep(THROTTLE)
                meta["audio_files"] = n_audio
                print(f"  [OK] audio×{n_audio}")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] 音频下载失败: {e}")
        elif args.audio:
            try:
                tree = api_json(f"/tracks/{w['id']}?v=2")
                auds = [(p, n.get("mediaDownloadUrl"), n.get("size") or 0)
                        for t, p, n in walk_tracks(tree if isinstance(tree, list) else [])
                        if t == "audio" and n.get("mediaDownloadUrl")]
                auds = [a for a in auds if a[2] <= args.max_audio_mb * 1024 * 1024]
                auds.sort(key=lambda a: a[2])
                if auds:
                    adir = wdir / "audio"
                    adir.mkdir(exist_ok=True)
                    _, url, _ = auds[0]
                    name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
                    data = api_get(url, binary=True)
                    (adir / name).write_bytes(data)
                    meta["audio_file"] = str((adir / name).relative_to(out_dir))
                    print(f"  [dl] audio/{name} ({len(data) // 1024}KB)")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] 音频下载失败: {e}")
        meta["subtitle_files"] = n_dl
        (wdir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] RJ{w['id']} -> {wdir}\n")


def login(name, password):
    """asmr.one 登录 -> JWT。凭据来自 seanime config.toml，严禁打印或写日志。"""
    body = json.dumps({"name": name, "password": password}).encode()
    req = urllib.request.Request(
        API + "/auth/me", data=body, method="POST",
        headers={"User-Agent": UA, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        resp = json.loads(r.read())
    token = resp.get("token") or resp.get("access_token")
    if not token:
        raise RuntimeError("登录响应无 token")
    return token


def read_seanime_credentials():
    """从 seanime 配置读 asmr.one 账号（config.toml [asmr] 段）。"""
    import tomllib
    cfg = Path("D:/ADLINK/Myproject/seanime/data/config.toml")
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    a = data.get("asmr") or {}
    if not a.get("name") or not a.get("password"):
        raise RuntimeError(f"{cfg} [asmr] 段缺 name/password")
    return a["name"], a["password"]


def cmd_favorites(args):
    """拉取账号书架全量(review 全状态,含想听/在听/听过)分页 + 逐部 workInfo 拿标题/tags，
    产出 favorites.json（与 inventory.json 的 works 同构，供 nightly 打分）。
    实测账号 marked=0、真实书架挂在 listening 等状态（2026-09-25），故默认不过滤。"""
    name, password = read_seanime_credentials()
    token = login(name, password)
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fav, page, total = [], 1, None
    while True:
        d = api_json(f"/review?order=updated_at&sort=desc&page={page}", headers=auth)
        batch = d.get("works", [])
        total = (d.get("pagination") or {}).get("totalCount", total)
        if not batch:
            break
        fav.extend(batch)
        if total is not None and len(fav) >= total:
            break
        page += 1
        time.sleep(THROTTLE)
    works = []
    for i, rv in enumerate(fav):
        wid = int(rv["id"])
        sid = (rv.get("source_id") or "").upper()
        if not sid.startswith("RJ"):
            sid = f"RJ{wid}"
        try:
            info = api_json(f"/workInfo/{wid}", headers=auth)
            title = info.get("title", "")
            tags = [t.get("name") for t in (info.get("tags") or []) if t.get("name")]
            works.append({
                "id": wid, "rj": sid, "title": title, "tags": tags,
                "progress": rv.get("progress", ""),
                "nsfw": bool(info.get("nsfw")),
                "dl_count": info.get("dl_count", 0),
                "duration_min": info.get("duration", 0),
                "has_subtitle": bool(info.get("has_subtitle")),
            })
            print(f"  [{i+1}/{len(fav)}] {sid} [{rv.get('progress','')}] "
                  f"{title[:38]} tags×{len(tags)}")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {sid} 详情失败: {e}")
        time.sleep(THROTTLE)
    out = out_dir / "favorites.json"
    out.write_text(json.dumps(
        {"total": total, "fetched": len(works), "works": works},
        ensure_ascii=False, indent=2), encoding="utf-8")
    n_sub = sum(1 for w in works if w["has_subtitle"])
    print(f"\n[OK] 书架 {len(works)}/{total} 部（带字幕标记 {n_sub} 部） -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="扫描搜索结果，按内容检测语言找中文字幕作品")
    s.add_argument("--pages", type=int, default=10)
    s.add_argument("--page-size", type=int, default=20)
    s.add_argument("--order", default="dl_count",
                   choices=["", "create_date", "dl_count", "price", "release",
                            "id", "rating"])
    s.add_argument("--normal", action="store_true", default=True,
                   help="标题粗筛排除成人向（默认开）")
    s.add_argument("--all", dest="normal", action="store_false",
                   help="关闭成人标题过滤")
    s.add_argument("--out", required=True)
    f = sub.add_parser("fetch", help="按清单下载入选作品")
    f.add_argument("--from-inv", required=True)
    f.add_argument("--limit", type=int, default=5)
    f.add_argument("--ids", default="",
                   help="逗号分隔的 work id，只下载这些（优先于 --limit）")
    f.add_argument("--audio", action="store_true", help="同时下载一个最小音频文件")
    f.add_argument("--audio-full", action="store_true",
                   help="全音轨下载（mp3/wav 去重，mp3 优先）")
    f.add_argument("--max-audio-mb", type=int, default=300)
    f.add_argument("--out", required=True)
    v = sub.add_parser("favorites", help="拉取账号收藏清单（含 tags）产出 favorites.json")
    v.add_argument("--out", required=True)
    args = ap.parse_args()
    {"scan": cmd_scan, "fetch": cmd_fetch, "favorites": cmd_favorites}[args.cmd](args)


if __name__ == "__main__":
    main()
