# tools/ — 离线评测与环境探测工具链

直播运行时只有 `live_sub.py` 一个入口；本目录是**离线**工具：喂音频文件跑完整管线出指标、
解析日志产报告、切评测素材、探测本机设备。任何时候都不被 `live_sub.py` import。

所有脚本的 `ROOT` 都锚定仓库根（`Path(__file__).resolve().parent.parent`），
从任意工作目录执行都可以：`.venv\Scripts\python.exe tools\xxx.py`。

## 脚本清单

| 脚本 | 运行条件 | 用途 | 一次性 |
| :--- | :--- | :--- | :--- |
| `run_custom_eval.py` | GPU+模型（几分钟） | dense/random 两场景全流程评测：起 live_sub 回放→解析 jsonl→汇总延迟/字幕数 | 常用 |
| `run_benchmark_suite.py` | GPU+模型 | Scene A/B 双策略（baseline vs adaptive）对比基准 | 常用 |
| `run_marathon_eval.py` | GPU+模型（十几分钟） | 30 分钟长时连续评测，看稳定性与积压 | 常用 |
| `run_soak_evaluation.py` | GPU+模型（十几分钟） | 15 分钟 soak 压测（内存/队列/降级路径） | 常用 |
| `build_teacher_groundtruth.py` | GPU+模型 | Teacher 真值：同一模型离线全篇 beam_size=5 转写（`--ja-only` 跳过旧译，参考译文统一走精译管线） | 常用 |
| `curate_gt.py` | 无（纯规则） | teacher 级 gt curation：U+FFFD 乱码 + avg_logprob<-1.15 双规则机械剔除，幂等 | 常用 |
| `build_dataset_manifest.py` | 无（纯规则） | 扫 dataset/*.gt_ja+ref_zh → `dataset/manifest.json` 标准索引（覆盖率/provenance/可用指标从实文件算） | 常用 |
| `evaluate_accuracy.py` | 只要日志（jiwer/rapidfuzz） | 实时 jsonl vs 真值 json → CER/吞字率 md 报告 | 常用 |
| `align_official_script.py` | 只要日志 | 官方台本 txt 对齐实时 jsonl → 真值 json（拟声词/低置信段标记 excluded 不参评） | 常用 |
| `subtitles_to_gt.py` | 无（纯解析） | srt/ass/ssa/vtt 字幕 → 金级真值 json：**时间戳权威，无需模糊匹配**；同行双语自动拆出 JA 真值 + 人工译文基准（human_zh.json） | 常用 |
| `asmrone_collect.py` | 无（纯 HTTP） | asmr.one 带字幕作品采集：扫描+按**内容**检测语言找中文字幕作品，下载字幕（LRC/VTT）与音频到语料库 | 常用 |
| `compare_references.py` | 无（纯计算） | 实时译文 vs 多参考（Sakura/云端/人工）相似度矩阵 + 参考层内部距离；剔除口径同 evaluate | 常用 |
| `build_quality_reference.py` | GPU+模型（约 10 秒） | 离线精译参考：同一 Sakura 贪心解码+长输出；`--gt` 模式从已有真值精译，脚本模式另产 zh_live 拆解差距 | 常用 |
| `make_md_table.py` | 只要日志 | 五份 jsonl → `logs/subtitles_table.md` 双语字幕总表 | 常用 |
| `dump_subtitles.py` | 只要日志 | 控制台打印 soak/Scene A/B 完整双语字幕 | 常用 |
| `dump_eval_report.py` | 只要日志 | `custom_scenes_full_result.json` → 评测报告 md | 常用 |
| `print_clean_results.py` | 只要日志 | 同上 json → 控制台干净打印 | 常用 |
| `inspect_quality_and_efficiency.py` | 只要日志 | soak/marathon/benchmark 三份日志的质量与效率巡检 | 常用 |
| `build_negative_samples.py` | 只要 benchmarks 音频（零 GPU） | 构建负样本对照段：用真 Segmenter 扫现有音频切“无字幕”30s 窗口 + 合成粉噪/类吟唱/静音 | 常用 |
| `check_negative_samples.py` | GPU+模型 | 对负样本段真跑完整管线，合格标准=**零字幕**（防幻觉对照组） | 常用 |
| `regroup.py` | 无（纯函数模块） | 日语句子重组规则（终助词/相槌+时间间隔硬约束），无 main，被 import | 保留 |
| `make_benchmarks.py` | D:\Downloads 原盘 | 生成 Scene A/B/C 素材（最早的蓝档案 Vol 8 切片） | 一次性 |
| `build_benchmarks.py` | D:\Downloads 原盘 | 生成 Scene A/B 素材 | 一次性 |
| `build_custom_benchmarks.py` | D:\Downloads 原盘 | 生成 dense/random 场景素材 | 一次性 |
| `build_marathon_audio.py` | D:\Downloads 原盘 | 拼三条音轨成马拉松长音频 | 一次性 |
| `build_soak_audio.py` | D:\Downloads 原盘 | 切 15 分钟 soak 素材 | 一次性 |
| `find_dense_and_random_audio.py` | D:\Downloads 原盘 | 扫全部音轨算人声密度，选评测素材用 | 一次性 |
| `probe_audio_devices.py` | 真实声卡+正在放音 | 逐个环回设备录 2 秒，探测哪个设备有信号 | 现场 |
| `check_monitors.py` | 真实显示器 | 枚举显示器坐标（配 `--screen` 选屏） | 现场 |
| `find_overlay_window.py` | 真实桌面 | 枚举可见窗口，确认字幕窗位置/置顶 | 现场 |
| `test_gui.py` | 真实桌面 | 弹 4 秒 tkinter 窗，验证 GUI 能显示 | 现场 |

## 复现与口径（重要：先读这段）

### 数据集：`dataset/`（2026-09-22 标准化产出）

准确度基准已收拢为标准化数据集：**5 个 scene × (gt_ja 真值 + ref_zh 精译参考)**，
索引 `dataset/manifest.json`（由 `build_dataset_manifest.py` 从实文件算出），
人类可读标准与层级定义在 `dataset/README.md`。音频留 `benchmarks/`
（DLsite 同人音轨有授权边界，不推远端）。

**两级真值，引用数字前必看层级**：

| 层级 | 场景 | 能声明 |
| :--- | :--- | :--- |
| gold（官方台本对齐） | scene_random_talk（21/26 段） | ASR 准确度 + MT 相似度 |
| teacher（离线全篇 beam=5） | a/b/c/dense（4/6、4/6、5/6、3/6 段） | **只有定性 diff + 延迟/效率**；CER/MT 数字不成立 |

teacher 级不产数字的实据：全篇无 VAD 解码会整段漏真实语音（scene_c 漏 3 处），
live 多识别的真实内容全被计成 Ins——`scene_c_pipeline_loss.md` 跑出
CER 97.52%，是参考漏检的产物不是管线错误。另：teacher 跨运行非确定
（CT2 GPU 归并），dataset 里的文件是钉死快照，换必须换名不覆盖。

### 当前有效口径（2026-09-22 v5，四件套同次配对）

`logs/benchmark_runs/accuracy_report_20260922_v5.md` 是目前唯一 (jsonl, gt, mt-ref, report)
四者同一次配对的有效报告（gt/ref 即 `dataset/scene_random_talk.*`）：

- **ASR**：字符准确率 80.96% / CER 19.04% / 吞字 16.87% / 幻觉 0.24%
  （真值 = 官方台本对齐，外部权威；**已实施双侧剔除**：5 个拟声/低置信真值段
  不参评，live 侧同步剔 5 段——v3/v4 的 83.71%/16.29%/14.43% 没消费 excluded
  字段，拟声段 official_ja 抄 live 自己的文本、双边自匹配灌水分母，偏乐观约 3 个点）
- **MT**：实时译文 vs 离线精译参考相似度 **71.08%**
  （`--mt-ref` = `dataset/scene_random_talk.ref_zh.json`）

**71.08% 的差距分解**（`benchmarks/quality_reference_20260922.json` 的 `zh_live`
参考把"识别损伤"从"翻译退化"里拆出来；均带同一套剔除）：

| 对比 | 相似度 | 含义 |
| :--- | :--- | :--- |
| 实时 vs 官方台本精译 (zh) | 71.08% | 端到端总差距 |
| 实时 vs 实时日文精译 (zh_live) | 91.28% | 纯翻译退化（喂同样的识别结果） |
| 参考侧 zh vs zh_live | 77.95% | 识别损伤往译文里的传导 |

即约 28.9 分的总差距里，约 20.2 分来自识别损伤往译文的传导、约 8.7 分来自
实时翻译的推理配置退化（temp=0.1 采样 / 80 token 截断）。**识别侧是绝对大头**，
与吞字 16.87% 互相印证。（2026-09-22 早前曾据未剔除数据得出"翻译侧更大"的
相反结论，那是拟声段垃圾译文污染的，已作废。）

历史报告只作存档：`accuracy_report_20260922_v4.md`（= v3 数字，未剔除）、
`v2` 的 39.62%（teacher 参考不干净）、`accuracy_report_official_mika.md` 的
"翻译对齐度 100%"（自比自恒等式，指标已废除）。

### 标准评测流程（按顺序）

```bash
# 1. 跑场景（会覆盖 custom_*.jsonl，跑完立刻快照！）
.venv\Scripts\python.exe tools\run_custom_eval.py

# 2. 真值：官方台本对齐本次 jsonl（只产日文真值，不再抄译文）
.venv\Scripts\python.exe tools\align_official_script.py ^
  --script benchmarks\script_mika_scene.txt ^
  --live logs\benchmark_runs\custom_random_talk.jsonl ^
  --out dataset\scene_random_talk.gt_ja.json

# 3. MT 参考：离线精译（贪心解码，约 10 秒，两次运行结果逐字节一致）
.venv\Scripts\python.exe tools\build_quality_reference.py ^
  --gt dataset\scene_random_talk.gt_ja.json ^
  --out dataset\scene_random_talk.ref_zh.json

# 4. 评分：MT 必须给 --mt-ref，否则该指标判"不可评"
.venv\Scripts\python.exe tools\evaluate_accuracy.py ^
  --live logs\benchmark_runs\custom_random_talk.jsonl ^
  --gt dataset\scene_random_talk.gt_ja.json ^
  --mt-ref dataset\scene_random_talk.ref_zh.json ^
  --out logs\benchmark_runs\accuracy_report_<日期>.md

# 5. 更新数据集索引，快照四件套（jsonl + gt + ref + report 一起归档，同名日期）
.venv\Scripts\python.exe tools\build_dataset_manifest.py
```

新场景入数据集：人工听写台本 → `align_official_script.py`（金级）；
或 `build_teacher_groundtruth.py --ja-only` → `curate_gt.py` →
`build_quality_reference.py --gt`（teacher 级，只配定性 diff）→
`build_dataset_manifest.py`。完整标准见 `dataset/README.md`。

`build_quality_reference.py` 与实时 `SakuraMT` 的差别只有推理参数：
贪心解码（temperature=0, top_k=1）、max_tokens=512、repeat_penalty=1.12、
GPU 全 offload；prompt 模板与模型同源，**不带前文上下文**——实测"参考上文"
格式下该量化模型会把上文一并译出、真实译文混在"文本："标记后，解析不可靠，
而上下文不是那三条实时约束之一，为它引入噪声不划算。参考里每段的 `zh_live`
（实时日文→同一套精译）可手动拆出"识别损伤 vs 翻译退化"，见上文差距分解表。

### 负样本对照（防幻觉）

```bash
.venv\Scripts\python.exe tools\build_negative_samples.py   # 构建 neg_*.wav（零 GPU）
.venv\Scripts\python.exe tools\check_negative_samples.py   # 真跑管线，合格=零字幕
```

全人声数据集考不出"该闭嘴时闭不闭嘴"——负样本段补上这个对照组。
`benchmarks/negative_samples.json` 是清单（6 个：3 真实窗口 + 粉噪/类吟唱/静音）。
2026-09-22 首跑 6/6 零字幕通过。

### 调参纪律

`docs/tuning_log.md`：阈值改动必须记录；**验收场景必须与调参场景不同源**；每次
评测跑完立即快照四件套。台账列了哪些参数是在现有场景上调出来的（即这些场景上的
数字偏乐观，幅度未知）。

### 确定性命令（输入不被覆盖，输出与历史逐字节一致）

```bash
.venv\Scripts\python.exe tools\make_md_table.py        # → logs/subtitles_table.md
.venv\Scripts\python.exe tools\dump_eval_report.py     # custom_scenes_full_result.json → md
```

`gt_teacher_*.json` 由 `build_teacher_groundtruth.py --audio <wav> --out <json>` 生成
（GPU 全篇转写+Sakura 离线翻译）。已知缺陷：时间戳不可靠、可能含退化段，只配做
MT 参考和参照系，不配做权威真值。2026-09-22 起 MT 参考首选
`build_quality_reference.py`（干净、确定、带 zh_live 可拆解差距），teacher 参考
退居备查。

## 整理记录（2026-09-22）

- 本目录 23 个脚本原散落在仓库根，未进 git；此次移入并统一 ROOT 锚定。
- `build_teacher_groundtruth.py` 删除了从未被调用的 `setup_cuda()`（CUDA DLL 注册由
  `import live_sub` → `livesub.config` 完成，功能重复），并加 `sys.path.insert` 保证
  移位后可 import；当日追加 `--ja-only`（跳过旧设置的 Sakura 译文，参考译文统一走
  精译管线）与 avg_logprob/no_speech_prob 质量字段。
- 已删除：`find_pid_windows.py`（PID 硬编码的一次性窗口调试）、
  `scratch/search_scripts.py`（Prowlarr 搜种脚本，内含明文 API key，按脱敏规矩不留）。
- 2026-09-22 数据集标准化：新增 `curate_gt.py`、`build_dataset_manifest.py`；
  `build_quality_reference.py` 加 `--gt` 模式；产出 `dataset/`（5 scene ×
  gt_ja+ref_zh + manifest + README）。金级（官方台本）只有 random 一个场景，
  其余 4 个为 teacher 级，数字指标不成立只配定性 diff——升级金级覆盖的唯一路径是
  人工听写台本（在库作品均不附带）。
