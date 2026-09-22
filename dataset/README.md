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

## 已知缺口

- 金级覆盖只有 1/5 场景——准确度声明的样本量本质上是一个场景，统计上单薄；
- teacher 级 4 个场景的真值与实时管线共享模型，只能测管线不能测模型；
- 所有场景都参与过阈值调整（见 `docs/tuning_log.md` 污染台账），
  在这些场景上的数字系统性偏乐观，幅度未知；
- 负样本只覆盖"无语音"，没有"非日语语音"（英语/噪音人声）对照。
