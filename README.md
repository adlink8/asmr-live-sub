# asmr-live-sub — ASMR 实时中文字幕系统

把日文 ASMR 音频/直播声音，实时转成低延迟的中文字幕。全链路本地推理（ASR + 翻译大模型都跑在自己机器上），零云依赖、零数据外传，消费级 8G 显卡即可流畅运行。

```text
系统回环捕获 ──► 16kHz 重采样 ──► 自适应分段 ──► ASR 识别 ──► 本地翻译 ──► 字幕窗口
   (WASAPI)        (重采样器)      (VAD off+静音挂起)  (CTranslate2)   (llama.cpp)    (tkinter)
```

## 特性亮点

- **端到端低延迟**：体感约 4.8s（段收口 3.6s + 识别翻译 0.77s），断句参数已在实测膝点，继续压延迟会显著牺牲召回。
- **全本地推理**：识别走 CTranslate2 量化模型，翻译走 llama.cpp 本地大模型（可切 NLLB / Qwen / Argos），断网可用。
- **确定性回放**：`--replay` 把任意音频变成可复现的测试输入（同一文件必得同一字幕序列），`--fast` 支持全速推流加速测评——整个优化过程都有可回归的尺子。
- **领域特调**：针对耳语/气声声学的参数取向（如 VAD 必须关闭，见踩坑实录），非通用语音方案直接套用。
- **测试先行**：141 项 pytest、标准化锚点轨道、负样本防幻觉门、双口径评测，见下文测试体系。

## 快速开始

### 环境要求

- Windows 10/11（音频捕获依赖 WASAPI 回环）
- Python 3.11+，CUDA 12 运行时（识别/翻译均走 GPU；纯 CPU 可跑但延迟大幅上升）
- NVIDIA 显卡 ≥ 8G 显存

### 模型放置

`models/` 目录：

| 子目录 | 内容 |
|---|---|
| `anime-whisper-ct2/` | 日文领域 CTranslate2 识别模型 |
| `sakura/` | 本地翻译大模型（llama.cpp GGUF） |
| `nllb/`、`qwen/`、`hub/` | 备选翻译模型与 HF 缓存 |

### 运行

```bash
# 枚举回环设备
python live_sub.py --list-devices

# 实时捕获（系统声音 → 中文字幕窗口）
python live_sub.py --model anime --mt sakura --strategy adaptive --max-s 5.0 --hang-s 2.0

# 离线转写单个文件（调试用）
python live_sub.py --source-audio demo.wav

# 确定性回放（全速、字幕序列落盘，用作回归基准）
python live_sub.py --source-audio demo.wav --replay --fast --log out.jsonl
```

## 实时参数速查

| 参数 | 生产值 | 说明 |
|---|---|---|
| `--strategy adaptive` | ✅ | 自适应断句（baseline/hardcap/adaptive 三选一） |
| `--max-s 5.0` | ✅ | 单段最大时长；实测 5.0 为延迟/召回膝点 |
| `--hang-s 2.0` | ✅ | 静音挂起断句；压到 1.0 缺字率 10.6%→18.3%，不划算 |
| `--seg-q-size 100` | 默认 | 分段队列容量；实时场景 100 足够，加速回放可调大 |
| `--fast` | 回放用 | 不按原速等待，全速推流 |

## 测试体系（质量门）

耳朵管"好不好用"，测试管"听测不出来的事"——翻译对不对、有没有漏、有没有退步。

- **单元/集成**：`pytest` 141 项全绿（`tests/`）。
- **锚点轨道**：每条取人工 cue 最密的 180s 标准窗，三个场景语义正确率 85.7% / 91.7% / 85.7%，覆盖率 92~97%；人工字幕作为"替你听日文"的基准真值。
- **负样本**：6 条纯音效/静音/无语音音频（`benchmarks/neg_*.wav`）——识别模型在无声段"造句"是最典型的幻觉，负样本把这类幻觉挡在版本门前。
- **双口径评测**：拟声/纯计数类整类剔出语义分母、单列"拦截数据"，禁止按结果挑数据。
- **LLM 语义裁判**：意思正确率由大模型盲判（不给参考译文防锚定），人工分歧样本走合议。

细则见 `docs/test_system_design.md`、`docs/tuning_log.md`。

## 数据与微调管线

`tools/` 内置一套从原料到训练集的完整流水线（通用领域语料工作流）：

1. **采集**：批量归档带字幕的作品为训练弹药——音频（转 mp3）+ 人工中文字幕 + 时间轴三件套，断点续扫、覆盖率门、磁盘水位门。
2. **双语对入库**：`ingest_gold_pairs.py` 从双语字幕按时间戳配对；`extract_bilingual_scripts.py` 从成对文本文档按语义向量配对（bge-m3 牵线 + 本地翻译模型验钞）。
3. **质检门（验钞门）**：本地翻译模型把日文句翻成中文，与人工中文做归一化字符相似度，分层直通/复核/隔离，全量留痕不丢弃。
4. **数据集标准**：gold / teacher 两级 + 多参考层；成色判定支持 LLM 裁判合议（盲判防锚定、毒对拦截率考核）。
5. **路线图**：采集量产 → 质检入库 → 识别模型微调（负样本治幻觉）→ 上线回灌数据飞轮。

编码防线：utf-8 → utf-16 → cp932 → gb18030 逐级回退（外部来源文本三种编码坑都实锤踩过），`errors=replace` 静默产乱码是禁止项。

## 领域踩坑实录（每条都付过学费）

1. **VAD 在气声耳语上是灾难**：默认 VAD 会把低响度气声段当噪声杀掉（实测段落损失 3 倍起），ASMR 场景必须 `vad_filter=False`。
2. **流式半截转写是另一个猜测，不是前缀**：截断音频喂识别模型，输出与整句真值的相似度仅 0.52~0.56——"边听边翻"的起翻优化因此判死。
3. **分段队列 newest-wins 驱逐有回放空洞**：容量触发时旧段被静默顶掉；实时路径碰不到，加速回放必须调大 `--seg-q-size`。
4. **Windows 三件套**：`taskkill` 斜杠转义不可靠（用 PowerShell `Stop-Process`）；`.cmd/.bat` 显式带 shell；文本一律 UTF-8 无 BOM + 多编码回退。
5. **并发分片必须"先取模定领地、再排除已完成"**，反了会跨片重复扫（实测重复 317 部）。
6. **CUDA DLL 注册**：`import live_sub` 会顺带注册 nvidia wheel 的 cublas/cudnn 目录，实验脚本直接 import faster_whisper 会报 dll 缺失。

更多见 `docs/issues_and_solutions.md`。

## 目录结构

```text
live_sub.py            # 入口（含 CUDA DLL 注册副作用）
livesub/               # 核心包：audio/pipeline/models/filters/gui/cli
tools/                 # 采集、数据入库、评测、实验脚本
tests/                 # pytest 测试（141 项）
benchmarks/            # 锚点真值与负样本
dataset/               # 评测用数据集标准
docs/                  # 模块地图/测试设计/调参日志/问题手册
models/                # 本地模型（不入库）
dataset_finetune/      # 训练弹药目录（数据素材按惯例不入库）
```

## 合规与边界

- 素材与数据一律留在本地磁盘，**不上传任何云**；API 密钥只放本机用户目录，严禁入仓库。
- 本仓库只包含代码与统计口径，**不包含、不链接、不分发任何第三方作品素材**；数据目录按惯例不入 git。
- 若对外发布：**模型权重可发布，数据集不发布**（权重是函数不是拷贝）；优先发 LoRA 而非合并权重；model card 不列任何作品清单。

## 文档索引

- `docs/module_map.md` — 模块地图
- `docs/test_system_design.md` — 测试体系总设计
- `docs/tuning_log.md` — 调参日志
- `docs/issues_and_solutions.md` — 问题与解决方案手册
