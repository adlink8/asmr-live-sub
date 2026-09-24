# dataset/ — 准确度基准数据集

本目录是**标准化数据集**：每个评测场景一份日文真值（`gt_ja`）+ 一份中文离线精译
参考（`ref_zh`），机器可读索引在 `manifest.json`（由
`tools/build_dataset_manifest.py` 从实际文件算出，数字不手抄）。

音频**不在**本目录：留在 `benchmarks/`（DLsite 同人作品音轨有授权边界，
任何情况下不推远端仓库；本目录只存元数据）。版权与用途见 manifest 的
`copyright` 字段。

## 一个场景长什么样

```
scene_random_talk.gt_ja.json    真值日文：分段 + t0/t1 + excluded/exclude_reason
scene_random_talk.ref_zh.json   参考中文：gt_ja 的 Sakura 贪心精译 + provenance
```

`gt_ja` 与 `ref_zh` 按 `seg_id` 对齐；`excluded` 段两侧同时不参评
（`evaluate_accuracy.py` 自动处理）。`ref_zh` 与任何一次 live 运行解耦——
它是场景属性，换管线和参数都能复用。

## 真值层级（引用数字前必读）

| 层级 | 制法 | 能声明什么 | 不能声明什么 |
| :--- | :--- | :--- | :--- |
| **gold** | 官方台本对齐（外部权威） | ASR 准确度（CER/吞字/幻觉）、MT 相似度 | 台本覆盖外的内容 |
| **teacher** | 离线全篇 beam=5 解码（无 VAD） | 人读逐句 diff；延迟/效率基准 | **任何数字指标**——CER 和 MT 相似度对它均不成立 |

teacher 级为什么不产数字（scene_c 实测证据）：全篇 beam=5 无 VAD 解码会
**整段漏掉真实语音**——scene_c 的 teacher 真值漏了 3 处真实台词
（"なるほど、いろいろとお仕事が…"等），而 live 管线把它们识别出来了。
拿这个参考算 CER，live 多识别出的**真实内容**全被计成幻觉插入：
scene_c 跑出 CER 97.52%（Ins 79.75%），这个数字是参考漏检的产物，
不是管线错误。漏检无界 → 任何汇总指标都无意义。

另三条硬约束（manifest 每个 scene 的 `caveats` 都写了）：

1. **时间戳不可靠**：t1≈t0+0.9s 与内容无关，逐句对照表仅供参考；
2. **跨运行非确定**：CT2 GPU 浮点归并顺序不定，两次解码的退化形态都不同——
   本目录的文件是**钉死的快照**，要换必须换文件名，不覆盖；
3. **非言语区污染**：吟唱/气声被离线解码成文字，而实时管线在这些区输出拟声词，
   两侧不同构。

curation 规则见 `tools/curate_gt.py`（U+FFFD 乱码 + `avg_logprob < -1.15`
低置信双规则，与实时管线 LOWCONF 同源）。

## 当前覆盖（2026-09-22，v1）

| 场景 | 音频来源 | 真值层级 | 保留/总段 | 可用指标 |
| :--- | :--- | :--- | :--- | :--- |
| scene_random_talk | 蔚蓝档案 圣园未夜轨 | **gold** | 21/26 | ASR 准确度 + MT 相似度 |
| scene_a | 蔚蓝档案 Vol 8 Iroha 03 | teacher | 4/6 | 定性 diff + 延迟/效率 |
| scene_b | 蔚蓝档案 Vol 8 Iroha 02 | teacher | 4/6 | 定性 diff + 延迟/效率 |
| scene_c | 蔚蓝档案 Vol 8 Iroha 04 | teacher | 5/6 | 定性 diff + 延迟/效率 |
| scene_dense_talk | hololive 特典杂谈 | teacher | 3/6 | 定性 diff + 延迟/效率 |

负样本对照（防幻觉）不在本目录：`benchmarks/negative_samples.json` +
6 个 `neg_*.wav`，合格标准=零字幕。

另有一条 **MT 端到端锚点轨道**（无 JA 真值，不做 ASR 评测）：
`scene_asmr299717_t01`，见下文专节；manifest 的 `mt_experiments` 字段索引它。

## 参考译文层（一个场景可有多份，manifest 的 refs 字段）

| 层 | 文件 | 性质 |
| :--- | :--- | :--- |
| Sakura 精译 | `<scene>.ref_zh.json` | 与实时同模型的贪心精译，确定性可复现；测"推理配置差距" |
| 云端模型 | `<scene>.ref_zh_stepflash.json` | step-3.7-flash 采样输出，不可复现快照；测"模型能力差距" |
| 人工译文 | `<scene>.human_zh.json` | 双语字幕拆出，**唯一绝对锚点**（种子素材到位后启用） |

**2026-09-22 首测的重要教训：相似度不能给参考排名。** step-3.7-flash 参考与实时译文
相似度只有 56.77%（Sakura 参考 71.08%，两参考彼此 56.64%），但逐段人工对照显示
step-flash **并不明显更好**：它把成语「烏の行水」（蜻蜓点水/速浴）误译为"乌鸦洗澡"
（Sakura 译对了），措辞更口语但有自己的错；而实时译文自己也有错（seg 8 主语"他们"）。
三个译文互有胜负——**没有人工锚点，参考层的质量排序在原则上看不出来**。
`tools/compare_references.py` 出距离矩阵，距离大不等于参考差，只说明风格分叉。

## 怎么用

```bash
# 金级场景：完整准确度声明
.venv\Scripts\python.exe tools\evaluate_accuracy.py ^
  --live logs\benchmark_runs\custom_random_talk.jsonl ^
  --gt dataset\scene_random_talk.gt_ja.json ^
  --mt-ref dataset\scene_random_talk.ref_zh.json ^
  --out logs\benchmark_runs\accuracy_report_<日期>.md

# teacher 级场景：只跑延迟/效率基准，或人读 diff 表——
# 它的 CER/MT 数字不成立（参考漏检无界，见上），别引用
.venv\Scripts\python.exe tools\evaluate_accuracy.py ^
  --live logs\benchmark_runs\baseline_scene_c.jsonl ^
  --gt dataset\scene_c.gt_ja.json ^
  --mt-ref dataset\scene_c.ref_zh.json ^
  --out logs\benchmark_runs\scene_c_diff_inspect.md
```

（该组合的实测产物 `logs/benchmark_runs/scene_c_pipeline_loss.md`：
CER 68.04%（Ins 52.51%）——live 多识别出的真实台词被全计成幻觉插入，
是参考漏检的产物，留作反面证据存档。）

## 怎么扩充

1. **新场景音频**：`tools/build_custom_benchmarks.py` 的规格加一条（180s）；
2. **金级真值**（唯一能升级准确度声明的路径）：人工听写台本 txt →
   `align_official_script.py` 对齐。在库作品均不附带台本，这是纯人工活，
   `script_mika_scene.txt` 是现成范例；
3. **teacher 级真值**：`build_teacher_groundtruth.py --ja-only` → `curate_gt.py`
   → `build_quality_reference.py --gt`；
4. 重跑 `build_dataset_manifest.py` 更新索引。

## 人工译文锚点场景（2026-09-22，asmr.one 路线）

`scene_asmr299717_t01`：RJ299717（助眠掏耳）track01，28 段。人工中文字幕
（汉化组 LRC，带时间戳）→ `scene_asmr299717_t01.human_zh.json`；
实时 ASR 日文 → `scene_asmr299717_t01.live_ja.json`（翻译实验输入，非真值）。
**该场景无 JA 真值，只做 MT 对标**。

该场景的三层度量（2026-09-22）：
1. **字符相似度**：Sakura 58.3% / flash 56.51%——风格+错位双重假象，此用途已废弃
2. **向量余弦（bge-small-zh，内容对齐后）**：Sakura 0.715 / flash 0.725——可规模化，
   绝对分未校准；未对齐时 0.45（错位污染），必须先对齐
3. **LLM 语义裁判（step-5-preview，意思正确率）**：**Sakura 96.4%（27/28）/
   flash 85.7%（24/28）**——准确但贵，n=28 读作估计

结论：实时翻译**意思正确率约 96%**，58.3% 约九成是度量假象；Sakura 在意思上
也赢 flash（域专才赢通用模型），换模型方向关闭，翻译侧无需优化。
裁判输入仍按索引配对（部分段人工参考错位），裁判靠 JA 仲裁大部分能识别
（seg27 主动标注了参考不对题），严谨用法是对齐后配对。

素材来源 `tools/asmrone_collect.py`（asmr.one API，按内容检测语言），
语料库在 `D:\Downloads\asmr-zh-corpus\`（版权内容，只留本地）。

## 锚点轨道（2026-09-23 标准化：全部 180s，与 5 个 ASR/MT 场景同规格）

asmr.one 扫描 20 部作品，13 部带人工中文字幕。已下载 4 部正常向作品
（RJ299717 / RJ401391 / RJ416816 / RJ416809）。**场景规格：每部作品取其人工
cue 最密的 180 秒窗口（窗口须完整落在音轨内；平手取最早），时间戳归零**——
时长一致是可比性的前提（早期版本用全轨，140s~1568s 横跨 11 倍，且长度直接
触发下述 seg_q 驱逐缺陷）。180s 下每场景段数 25~34 个，远低于 seg_q(100)
容量线，回放 harness 缺陷被规格天然规避。

| 场景 | 作品 | 人工 cue | 参评配对 | missing | **语义正确率** | 拦截数据 |
|---|---|---|---|---|---|---|
| scene_asmr299717_t02 | RJ299717 05_お茶の時間 | 23 | 22 | 1 | **85.7%**（n=21） | 1 拟声 |
| scene_asmr416809_t01 | RJ416809 泳装应援 | 26 | 24 | 2 | **91.7%**（n=24） | 0 |
| scene_asmr416816_t01 | RJ416816 安眠诱导 | 32 | 31 | 1 | **85.7%**（n=21） | 10 计数 |
| ~~scene_asmr299717_t01~~ | RJ299717 track01（全轨 140s） | 28 | 28 | 0 | 96.4%（legacy，勿与新场景对比） | 0 |

`scene_asmr299717_t01` 是 2026-09-22 的首个锚点（全轨 140s，已入 git，
按"钉死快照不覆盖"纪律保留作 legacy 参照）；其 96.4% 与上表三个数**不可比**
（时长 140s vs 180s、无 kind 分类）。跨场景比较只用标准化三个。

**三个标准化场景的结论**：实时翻译在人工台词上的意思正确率 **85.7%~91.7%**，
人工行覆盖率 92%~97%。partial 主因是 ASR 乱识直译（エムデン→"M电/M电话"、
ふぁいと→"好"、イジュイテ→"把他们"），与 v5 口径"识别损伤是 MT 差距大头"
互相印证；翻译本身（喂对 JA 时）基本可靠。

**双口径（引用数字必须带口径）**：拟声词/纯计数行整类移出语义分母（连判对的
一起剔——只剔失败的会人为灌水，与 gold 场景"双侧剔除"同一纪律），单列为
**拦截数据**：过滤器/ASR 在非言语区的漏放质量（该拦没拦），不是翻译质量。
`semantic_judge.json` 每条 verdict 带 `kind`（dialogue/sound/countdown），
`summary` 是全量口径、`summary_dialogue` 是语义口径。

**RJ401391 已判定不可用并踢出**（产物已删）：该作品 Track01 的 LRC 是叙事
台词译文，但音频 520s 里 ASR 只能听到非言语声响，全轨无可辨认台词——LRC 与
音频不对应（疑为视频版字幕配音频版音轨）。待查：下载其他轨验证对应关系。

**回放 harness 缺陷（全轨跑时踩到，180s 规格下规避，但 soak/marathon 仍受害）**：
1. **seg_q 驱逐**：fast 推流约 1 秒涌出全部段（RJ416809 全轨产出 283 段），
   seg_q(100) 用 `_put_or_bump`（newest-wins），超容的最旧段（开场）在 asr_loop
   够到前被逐。隔离实测复现：无消费者时幸存段仅覆盖 993.5~1555s。此即
   tests/README 挂了很久的"回放背压损耗"；seg_id 连续性检验是 red herring
   （seg_id 由 asr_loop 识别时才分配，丢多少都连续）。
2. **result_q 显示丢失**（`docs/issues_and_solutions.md` 问题八）：推流结束后
   `seg_q.put(None)` 阻塞把排水循环推迟数十秒，期间 result_q(50) 无消费者，
   最旧字幕被逐（全轨 416809 实测 183 译 → 145 显示）。
两条都只影响回放/评测路径，不影响 live 生产路径。修复候选见问题八。

**素材 staging 状态（用户决策 2026-09-23）**：语料库另有 9 部成人向作品
已下载字幕+音频（4 部因最小音轨超 300MB 上限仅拿到字幕），**只进语料库、
不进测试集**；待正常向作品的"测试+语义对比"全流程验收完成后，再决定是否纳入。


## 已知缺口

- 金级覆盖只有 1/5 场景——准确度声明的样本量本质上是一个场景，统计上单薄；
- teacher 级 4 个场景的真值与实时管线共享模型，只能测管线不能测模型；
- 所有场景都参与过阈值调整（见 `docs/tuning_log.md` 污染台账），
  在这些场景上的数字系统性偏乐观，幅度未知；
- 负样本只覆盖"无语音"，没有"非日语语音"（英语/噪音人声）对照。
