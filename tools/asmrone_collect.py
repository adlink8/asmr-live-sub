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

# 翻译系社团（汉化组）：registry 全池里自己不创作、专做别人作品翻译版的社团。
# 探矿实证（prospect_gold 2026-09-26）：中文字幕 10/10 全中（覆盖率 73~99%）但
# 日文字幕 0/10——是 zh 侧人工真值的"半金矿"，nightly 以限额穿插优先采集。
TRANSLATION_CIRCLES = {
    "MYHONYAKU", "无糖可乐", "HerbPear-translations-",
    "Dear Violin(特典音轨本来就是必翻项)", "暁の繁体翻訳", "HTCHEN翻譯",
    "结系汉化组", "毒刺翻譯", "大家一起来翻译", "漁貓翻譯組(売り子再更新)",
}

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


def sub_max_ts(raw):
    """字幕文本里的最大时间戳（秒）。LRC [mm:ss.xx] / VTT/SRT HH:MM:SS,ms --> 两种都认，
    无时间戳的纯文本返回 0。"""
    mx = 0.0
    for ln in raw.splitlines():
        ln = ln.strip()
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]", ln)
        if m:
            mx = max(mx, int(m.group(1)) * 60 + float(m.group(2)))
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", ln)
        if m:
            mx = max(mx, int(m.group(1)) * 3600 + int(m.group(2)) * 60
                     + int(m.group(3)) + int(m.group(4)) / 1000)
    return mx


def classify_work(wid, title):
    """拉音轨树，逐文本文件下载判语言。无中文字幕 -> None。
    同时按字幕最大时间戳/音轨时长算翻译覆盖率（时长加权，mp3/wav 去重后口径），
    覆盖率在 scan 阶段就得出——翻译率太低的作品在筛选时直接抛弃，不浪费 fetch/采集。"""
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
    # 翻译覆盖率：唯一化音轨（mp3 优先）作分母；同轨多个中文译本取最大时间戳
    uniq = dedup_audio(audio)
    dur_by_stem = {}
    for p, _, _ in uniq:
        stem = Path(p).stem.lower()
        if stem not in dur_by_stem:
            dur_by_stem[stem] = _node_duration(flat, p)
    cov_by_stem = {}
    for p, url in zh_files:
        stem = Path(p).stem.lower()
        if stem not in dur_by_stem or not dur_by_stem[stem]:
            continue
        try:
            raw = api_get(url, binary=True).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        ts = sub_max_ts(raw)
        cov = min(1.0, ts / dur_by_stem[stem]) if ts else 0.0
        cov_by_stem[stem] = max(cov_by_stem.get(stem, 0.0), cov)
    total_dur = sum(dur_by_stem.values()) or 1
    covered = sum(dur_by_stem[s] * c for s, c in cov_by_stem.items())
    # 金级就绪：ja 字幕文件 stem 对得上音轨且带时间戳（排除 readme/凑数 txt）
    gold_ready = 0
    for p, url in ja_files:
        stem = Path(p).stem.lower()
        if stem in dur_by_stem:
            try:
                ts = sub_max_ts(api_get(url, binary=True)
                                .decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                continue
            if ts > 0:
                gold_ready += 1
                break
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
        # 翻译覆盖率 0~1（时长加权）；纯文本无时间戳的译本计 0
        "sub_coverage": round(covered / total_dur, 3),
        # 双语字幕金级就绪（ja 转写 stem 匹配+带时间戳）
        "gold_ready": gold_ready > 0,
    }


def _node_duration(flat, path):
    for t, p, n in flat:
        if t == "audio" and p == path:
            return n.get("duration") or 0
    return 0


def cmd_scan(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    found, scanned, seen = [], 0, set()
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    inv = out_dir / "inventory.json"
    for order in orders:
        for page in range(1, args.pages + 1):
            d = api_json(f"/search/{page}?subtitle=1&pageSize={args.page_size}"
                         f"&order={order}")
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
                        "duration_sec": w.get("duration", 0),  # asmr.one API 的 duration 单位是秒
                        # 官方 tag 全集（[{id,name}] -> 名字列表），标签补全打分的原料
                        "tags": [t.get("name") for t in (w.get("tags") or []) if t.get("name")],
                    })
                    found.append(info)
                    print(f"  [hit] {wid} zh×{len(info['zh_urls'])} "
                          f"ja×{info['ja_sub_count']} audio×{info['audio_count']} "
                          f"{title[:38]}")
                time.sleep(THROTTLE)
            # 增量落盘：长扫描中断不清零
            inv.write_text(json.dumps(
                {"scanned": scanned, "with_zh_subtitle": len(found),
                 "with_ja_sub": sum(1 for f in found if f["ja_sub_count"]),
                 "works": found}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ja = sum(1 for f in found if f["ja_sub_count"])
    print(f"\n[OK] 扫描 {scanned} 部（去重后），带中文字幕 {len(found)} 部，"
          f"其中同日文字幕 {n_ja} 部")
    print(f"     清单: {inv}")


def cmd_registry(args):
    """全池轻量登记：走 /api/works 端点翻页（subtitle=1 全池 7930 部≈40 页）。

    端点探底结论（2026-09-26）：/api/search 的 page/keyword/tag 参数全部失效
    （page 恒返回第一页），只有 order×sort 组合可扩面，天花板 1932 部；社区
    下载器（fireinrain/asmr-downloader 的 api.http）揭晓正解是 **/api/works**
    端点——page 翻页有效、pageSize 到 200 有效、subtitle 过滤有效。

    产出 registry_full.json，独立于夜间任务的 inventory.json（后者带
    sub_coverage 字段且被覆盖率门依赖，混入无此字段的全池数据会坏门）。
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reg_path = out / "registry_full.json"
    works, seen = [], set()
    if reg_path.exists():
        works = json.loads(reg_path.read_text(encoding="utf-8")).get("works", [])
        seen = {w["id"] for w in works}
        print(f"[resume] 已有 {len(works)} 部，增量续扫")

    order, _, sort = args.orders.partition(":")
    sort = sort or "desc"
    page, total = 1, None
    while page <= args.max_pages:
        d = api_json(f"/works?order={order}&sort={sort}&page={page}"
                     f"&pageSize={args.page_size}&subtitle=1")
        batch = d.get("works", [])
        total = (d.get("pagination") or {}).get("totalCount") or total
        fresh = 0
        for w in batch:
            wid = w["id"]
            if wid in seen:
                continue
            seen.add(wid)
            fresh += 1
            ti = w.get("translation_info") or {}
            works.append({
                "id": wid,
                "title": w.get("title", ""),
                "nsfw": bool(w.get("nsfw")),
                "release": w.get("release", ""),
                "rating": w.get("rate_average_2dp", 0),
                "rate_count": w.get("rate_count", 0),
                "dl_count": w.get("dl_count", 0),
                "duration_sec": w.get("duration", 0),
                "circle": w.get("circle"),
                "vas": [v.get("name") for v in (w.get("vas") or []) if v.get("name")],
                "tags": [t.get("name") for t in (w.get("tags") or []) if t.get("name")],
                # 官方语言版本（JPN/CHI_HANS/...）
                "langs": [e.get("lang") if isinstance(e, dict) else e
                          for e in (w.get("language_editions") or [])
                          if (e.get("lang") if isinstance(e, dict) else e)],
                # 官方翻译谱系：is_child=翻译版, is_volunteer=志愿者翻译, lang=译入语
                "translation": {k: ti.get(k) for k in
                                ("lang", "is_child", "is_volunteer", "is_original")
                                if ti.get(k) is not None} or None,
            })
        print(f"[pg {page}] +{fresh}/{len(batch)} 累计 {len(works)}/{total}")
        reg_path.write_text(json.dumps(
            {"total": total, "works": works}, ensure_ascii=False),
            encoding="utf-8")
        if not batch:
            break
        page += 1
        time.sleep(THROTTLE)
    print(f"\n[OK] 全池登记 {len(works)} 部 -> {reg_path}")


def cmd_check(args):
    """单作品现场检测：拉音轨树下字幕判语言（classify_work），产出 fetch
    兼容的单作品清单（--from-inv 可直接喂）。给 nightly 的翻译社团队列做
    按需检测用——registry 只有元数据，字幕 URL 要现场拉树才有。
    exit 0=有中文字幕（清单已写）；1=无中文字幕。"""
    info = classify_work(int(args.id), args.title or "")
    if info is None:
        print(f"[check] RJ{args.id} 无中文字幕")
        raise SystemExit(1)
    info.setdefault("tags", [t.strip() for t in (args.tags or "").split(",") if t.strip()])
    Path(args.out).write_text(
        json.dumps({"works": [info]}, ensure_ascii=False), encoding="utf-8")
    print(f"[check] RJ{args.id} zh×{len(info['zh_urls'])} "
          f"ja×{info['ja_sub_count']} 覆盖率={info['sub_coverage']:.0%} "
          f"清单: {args.out}")


def cmd_fetch(args):
    inv = json.loads(Path(args.from_inv).read_text(encoding="utf-8"))
    works = [w for w in inv["works"] if w.get("zh_urls")]
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        works = [w for w in works if w["id"] in want]
    # 优先：有日文字幕的（JA 金级+ZH 人工双全），再按下载量（.get 兜底：
    # check 产出的单作品清单来自 classify_work，不带 dl_count）
    works.sort(key=lambda w: (w.get("ja_sub_count", 0) > 0, w.get("dl_count", 0)),
               reverse=True)
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


def _work_brief(info, progress=""):
    """rawWork 同构条目 -> 与 inventory works 同构的精简条目。"""
    return {
        "id": int(info["id"]),
        "title": info.get("title", ""),
        "tags": [t.get("name") for t in (info.get("tags") or []) if t.get("name")],
        "progress": progress,
        "nsfw": bool(info.get("nsfw")),
        "dl_count": info.get("dl_count", 0),
        "duration_sec": info.get("duration", 0),
        "has_subtitle": bool(info.get("has_subtitle")),
    }


def cmd_favorites(args):
    """拉取账号书架全量(review 全状态) + 自定义分组(get-playlists，系统列表滤除)，
    产出 favorites.json：works=书架条目，groups=[{name, works=[...]}]（组内作品与
    inventory works 同构、自带 tags）。分组名在 nightly 里当作用户自定义 tag 计入基线。"""
    name, password = read_seanime_credentials()
    token = login(name, password)
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 书架（review 全状态分页）
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
        try:
            w = _work_brief(api_json(f"/workInfo/{int(rv['id'])}", headers=auth),
                            rv.get("progress", ""))
            works.append(w)
            print(f"  [书架 {i+1}/{len(fav)}] RJ{w['id']} [{w['progress']}] "
                  f"{w['title'][:36]} tags×{len(w['tags'])}")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] RJ{rv['id']} 详情失败: {e}")
        time.sleep(THROTTLE)

    # 自定义分组（系统 __SYS_ 列表滤除；组内 works 与 rawWork 同构，免二次请求）
    groups = []
    page = 1
    pls = []
    while True:
        d = api_json(f"/playlist/get-playlists?page={page}", headers=auth)
        batch = d.get("playlists", [])
        if not batch:
            break
        pls.extend(batch)
        page += 1
        time.sleep(THROTTLE)
    for i, pl in enumerate(pls):
        if (pl.get("name") or "").startswith("__SYS_"):
            continue
        gw, page2 = [], 1
        while True:
            d = api_json(f"/playlist/get-playlist-works?id={pl['id']}"
                         f"&page={page2}&pageSize=20", headers=auth)
            batch = d.get("works", [])
            if not batch:
                break
            gw.extend(batch)
            page2 += 1
            time.sleep(THROTTLE)
        briefs = [_work_brief(w) for w in gw]
        groups.append({"id": pl["id"], "name": pl.get("name", ""),
                       "works": briefs})
        print(f"  [分组 {i+1}] {pl.get('name','')} {len(briefs)} 部")

    out = out_dir / "favorites.json"
    out.write_text(json.dumps(
        {"total": total, "fetched": len(works), "works": works, "groups": groups,
         "tag_vocab": api_json("/tags", headers=auth)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    n_sub = sum(1 for w in works if w["has_subtitle"])
    print(f"\n[OK] 书架 {len(works)}/{total} 部（带字幕 {n_sub}）+ "
          f"分组 {len(groups)} 个 + 官方 tag 全集 -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="扫描搜索结果，按内容检测语言找中文字幕作品")
    s.add_argument("--pages", type=int, default=5, help="每个排序方向扫的页数")
    s.add_argument("--page-size", type=int, default=100)
    s.add_argument("--orders", default="dl_count,create_date,rating",
                   help="逗号分隔的多排序轮扫（去重后并集），覆盖不同头部作品")
    s.add_argument("--normal", action="store_true", default=False,
                   help="开启成人标题过滤（2026-09-25 用户下令默认拔除：成人/正常声学上无本质区别）")
    s.add_argument("--all", dest="normal", action="store_false",
                   help="关闭成人标题过滤（现已是默认）")
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
    c = sub.add_parser("check", help="单作品现场检测（classify_work），产出 fetch 兼容清单")
    c.add_argument("--id", required=True)
    c.add_argument("--title", default="")
    c.add_argument("--tags", default="", help="逗号分隔 tags（从 registry 透传，仅早报展示用）")
    c.add_argument("--out", required=True)
    r = sub.add_parser("registry", help="全池轻量登记（/api/works 翻页，7930 部≈40 页）")
    r.add_argument("--page-size", type=int, default=200)
    r.add_argument("--orders", default="create_date:desc",
                   help="order:sort（/api/works 翻页有效，单组合即可拉全池）")
    r.add_argument("--max-pages", type=int, default=60, help="保险上限")
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"scan": cmd_scan, "fetch": cmd_fetch, "favorites": cmd_favorites,
     "registry": cmd_registry, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
