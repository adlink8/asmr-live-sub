# 模块分类与架构图（2026-09-22 拆分）

`live_sub.py` 原为约 1200 行单文件。2026-09-22 按功能边界拆为 `livesub/` 包，
**入口与 CLI 契约完全不变**（`start.bat`、`--help` 参数、日志格式、JSONL 字段全部一致），
`live_sub.py` 保留为薄入口 + 兼容层。

## 1. 数据流与线程模型

```
                 ┌──────────────── live capture (cli.py) ────────────────┐
                 │                                                       │
  WASAPI 环回 ──► │ PortAudio callback ─► FrameBuffer ─► feeder(100ms)   │
  (系统声音)      │        (audio.py)                      │              │
                 │                                        ▼              │
                 │                              Segmenter.feed() 能量门切块 │
                 │                                        │ (audio16,t0,t1)│
                 │                                        ▼              │
                 │                              seg_q(100)               │
                 └────────────────────────────────────────│───────────────┘
                                                          ▼
                              asr_loop (pipeline.py) ── faster-whisper ja 识别
                                        │ drop_reason 文本过滤 (filters.py)
                                        ▼ mt_q(2) / late_q(200)
                              mt_loop (pipeline.py) ── Sakura/NLLB/Qwen 翻译
                                        │ is_refusal 拒答过滤
                                        ▼ result_q(50)
                              SubtitleWindow.poll 120ms (gui.py) ── 置顶字幕
```

线程：`feeder` / `asr_loop` / `mt_loop` 三个 daemon + tkinter 主线程。
离线路径（`--source-audio [--replay]`）复用同一条管线，只是把采集段换成文件解码段。

**降级策略（设计如此，触发计数见日志）**：`asr_loop` 批量只识别最新段、其余转 `late_q`；
`mt_loop` 积压时只留最新。`late_q` 内容只进日志不上屏——字幕永远显示最新一行。

## 2. 模块职责与依赖

| 模块 | 职责 | 关键导出 | 依赖 |
|---|---|---|---|
| `config.py` | 路径、环境变量、调参常量、CUDA DLL 注册 | `ROOT` `TARGET_SR` `MIN_SEG_S` `LOWCONF_*` `HAVE_CUDA_DLLS` | 无（stdlib） |
| `logging.py` | 控制台/文件行日志 + JSONL 事件 trace | `LineLog` `WorkLog` | 无（stdlib） |
| `filters.py` | 文本过滤层全部规则 | `drop_reason` `is_kana_loop` `is_refusal` `strip_punct` | config |
| `audio.py` | 帧缓冲、重采样、能量门 Segmenter、设备枚举 | `Segmenter` `FrameBuffer` `resample_to_16k` `enumerate_loopbacks` | config |
| `models.py` | ASR 加载 + 三个 MT 后端 | `load_model` `load_translator` `SakuraMT` `NllbMT` `QwenMT` | config |
| `pipeline.py` | asr/mt 两个线程循环 + 单段 decode | `asr_loop` `mt_loop` `asr_decode` `drain_all` | config, filters, audio, logging |
| `gui.py` | tkinter 置顶字幕窗、多屏定位 | `SubtitleWindow` `choose_monitor` | 无（stdlib；tkinter 延迟到 `__init__` 内导入） |
| `cli.py` | argparse、日志路径、队列装配、三种模式 | `main` | 以上全部 |
| `__init__.py` | 包公开 API（精选 25 个名字） | 见 `__all__` | 以上全部 |

**分层规则**：`config → logging → filters → audio → models → pipeline → gui → cli`，
只允许单向依赖；`cli` 是唯一的组装点。重库（torch/faster_whisper/llama_cpp/
ctranslate2/av/pyaudiowpatch/tkinter）全部延迟到函数内导入，
所以 `import livesub` 不加载任何模型、不碰 GPU（CUDA DLL 注册除外，那是config的副业）。

## 3. 兼容层（live_sub.py）

保留原因有两个，都不是历史惯性：

1. `start.bat` 调用 `python live_sub.py`；
2. `tools/build_teacher_groundtruth.py` `import live_sub` 直接用
   `live_sub.load_model(args)` / `live_sub.SakuraMT(...)`，
   且依赖"导入即注册 CUDA DLL 路径"这一副作用（否则 experiment 脚本报
   `cublas64_12.dll is not found`）。该脚本 2026-09-22 已从根部移入
   `tools/`，自带的 `setup_cuda()` 死代码随之删除（注册由 import 完成），
   移位后可 import 靠 `sys.path.insert(0, ROOT)`。

shim 只做 re-export，不含逻辑。新代码请 `from livesub.xxx import yyy`。

## 4. 深模块与接口评估（2026-09-22 自检）

**深模块（接口窄、实现厚）**：

- `filters.py`：4 个函数封装全部幻觉/循环/置信度规则，阈值依据实测分布
  （正常段 logprob -0.54~-0.61 vs 幻觉 <-1.15），调用方不需要知道任何正则。
- `Segmenter`：一个 `feed(chunk)` 入口隐藏双模能量门、三种切块策略、
  静音垫剪裁、段边界追踪；调用方只收 `(audio16, t_start, t_end)` 三元组。
- `models.py` 的三个 MT 后端：统一 `translate(text) -> str` + `last_stats` 协议，
  管线只认协议，不认后端；换 MT 是构造函数选择，零管线改动。

**浅但合理的模块**：

- `cli.py` 约 200 行是组装根（argparse + 三种模式分支），天然是应用里最"浅"的
  模块，可接受；其中 live 采集段（callback/feeder/超时线程）如继续膨胀，
  下一步应抽 `capture.py`，现在规模未到那个临界点。
- `gui.py` 的 `choose_monitor` 已从构造函数抽出为纯函数（可单测）；
  Win32 样式操作仍在 `__init__`，属于平台胶水，不再拆。

**接口面**：`__init__` 只 re-export 25 个跨模块稳定名字；内部 helper
（`_put_or_bump`、`_update_gate`、`_add_cuda_dlls`）以下划线约定私有，
需要时从子模块显式导入，不进公开 API。

## 5. 重复造轮子审查（生态优先）

逐项核对过，结论：**只有一处值得讨论，其余不造轮子**。

| 位置 | 现状 | 成熟替代 | 结论 |
|---|---|---|---|
| `audio.resample_to_16k` | `np.interp` 线性插值 6 行 | `scipy.signal.resample_poly`（多相滤波，抗混叠）；`samplerate`(secretrabbit) | **保留**：48k→16k 的线性插值有混叠失真，但 Whisper 吃 log-Mel 谱、对该失真稳健；换 scipy 要为 6 行函数引入 ~40MB 依赖。若将来换更高采样率源或换 ASR 模型，这里优先换 `resample_poly`。已在代码注释标注。 |
| `FrameBuffer` | bytearray+Lock 15 行 | 无（stdlib 没有音频环形缓冲） | 不造轮子，是必要胶水 |
| `WorkLog` | append-only JSONL + Lock | stdlib `logging`+JSON formatter | 不换：事件是 (kind, **kwargs) 扁平结构，现实现比配置 logging 更直白；且 analyze 脚本直接读 JSONL，格式是契约不能动 |
| `Segmenter`/能量门/过滤规则 | 手写 | WhisperJAV 是参考项目不是可调用的库 | 领域逻辑，无从复用 |
| VAD（layer 2） | 不自己写，走 faster-whisper 的 Silero | — | 已用生态（且实测在 ASMR 上负收益，默认 layer 1） |

## 6. 拆分与测试过程中发现并修复的潜在 bug

1. **`_force_topmost` 从未生效**（拆分时发现）：原文件无模块级 `import ctypes`，方法内
   `ctypes.windll` 抛 NameError 被裸 `except` 吞掉 → SetWindowPos(HWND_TOPMOST)
   一直没跑，实际只有 tkinter `-topmost` 在撑。已在 `livesub/gui.py` 模块级补 import。
2. **`Segmenter.feed()` 绝对位置算错**（写测试时发现，最严重）：`_total_samples` 到
   `feed()` 末尾才更新、`_pending` 末尾才裁剪，循环里 `frame_abs = _total_samples + pos`
   没减掉上轮遗留的 `carry`——**段起点 `t_start` 最多虚高一个帧长（0.51s）**，
   `t_end` 用 feed 入口时的旧计数器（单次大数组喂入时恒为 0）。
   全部调优依赖的段边界日志一直是错的。已修：`carry` 修正 + `_emit(end_abs)` 显式收尾位置。
   注意：修复后 `curr_seg_s`（adaptive/hardcap 的墙钟判据）也变准了，切块时机会有微小变化。
3. **去重比较不同口径**：`last_ja` 窗口存的是去标点核心文本，比较却用带标点的 `ja`——
   「ありがとう」vs「ありがとう。」漏判 repeat。已改为同口径比较。
4. **关机哨兵吞掉最后一条字幕**：`mt_loop` live 分支中 None 紧跟在 live 项后入队时，
   旧实现把该 live 项丢进 `late_q` 后直接 break——最后一条 live 字幕既不翻译也不上屏
   （违反 newest-wans 本意的静默丢失）。已修：先翻译完最新项再退。

另有两个**锁定未改**的已知歧义（测试钉住现状，决策见 `tests/README.md`）：
`--mt none` 实际不上屏（帮助文本说 ja only）；replay 全速推流时 `seg_q(100)` 满后
newest-wins 会静默丢段（长音频评测保真度待实测）。

## 7. 验证记录

- 2026-09-22 拆分日：`py_compile` 全过 / `import live_sub` 兼容名全过 / `--help` 一致 /
  `--source-audio --replay --fast` 全链路出 2 条字幕 / 过滤器判定回归一致。
- 2026-09-22 测试日：`pytest` **141 个全绿**（约 4 秒，无需 GPU/模型/音频设备）；
  三个 bug 修复后生产配置离线回放复测仍出 2 条字幕，无回归。
- 测试体系四层地图、stub 契约、已知歧义清单见 [`../tests/README.md`](../tests/README.md)。
