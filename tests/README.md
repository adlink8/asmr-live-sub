# 测试体系（2026-09-22 建）

四层测试，全部用项目自带 venv 跑，**不需要 GPU、不需要模型权重、不需要音频设备**——
模型边界一律用 duck-typed stub（见 `conftest.py`），跑的是生产代码路径。

```bash
.venv\Scripts\python.exe -m pytest            # 全量（141 个，约 4 秒）
.venv\Scripts\python.exe -m pytest tests/test_segmenter.py -v
```

## 四层地图

| 层 | 文件 | 测什么 | 关键手法 |
|---|---|---|---|
| **单元** | `test_filters.py` | 每条过滤规则 + 调优阈值边界（low-conf -1.15/4.0s、no-speech 0.6/2.5s、compression 2.4、sound-only 24 字窗、`はい` 必须活） | 纯函数，参数化 |
| | `test_logging.py` | LineLog 双写、WorkLog JSONL  schema、8 线程并发不撕行 | tmp_path + capsys |
| | `test_audio.py` | FrameBuffer 溢出丢旧、重采样恒等/减半、`_put_or_bump` 最新胜出 | 纯逻辑 |
| **单元+回归** | `test_segmenter.py` | 切块语义全锁定：MIN_SEG_S 门槛、hang_frames 取整、尾部静音剪裁、15 帧配额名不副实、adaptive 阶梯、hardcap 墙钟、gate 双模滞回、段边界连续性 | 合成帧模式串（`v`/`s`） |
| | `test_pipeline.py` | asr_decode 分层/layer2 VAD 参数/去重窗口（5 条淘汰）、asr_loop live 降级 vs replay 保序、mt_loop 拒答丢弃/积压最新胜出/异常不死/关机哨兵 | stub 模型 + 真线程真队列 |
| **集成** | `test_integration.py` | 真实 Segmenter→asr_loop→mt_loop→result_q 全链路（队列尺寸同生产）、采集管路 FrameBuffer+feeder 与直喂等价、10 句回放零丢失、CLI `--help` 子进程、新解释器 compat import | stub 只在模型边界 |
| **契约** | `test_contracts.py` | JSONL 事件字段 schema（seg-drop/gate/asr/asr-batch/mt 的必填键——评测脚本按名读）、队列 item 形状（段三元组/mt 三元组/result 三元组/旧两元组兼容）、MT 后端协议（translate+last_stats 双成员即可互换）、`live_sub` 兼容面 25 个名字、layer≥2 警告 | 形状断言，不测行为 |

## 测试挖出并已修复的 bug（修复+回归锁定）

1. **`Segmenter.feed()` 绝对位置算错**（最严重）：`_total_samples` 到 `feed()` 末尾才更新、
   `_pending` 末尾才裁剪，循环里 `frame_abs = _total_samples + pos` 没减掉上轮遗留的
   `carry`——**段起点 t_start 最多虚高一个帧长（0.51s）**；`t_end` 用 feed 入口旧计数器
   （单次大数组喂入时恒为 0）。全部调优依赖的段边界一直是错的。已修（`carry` 修正 +
   `_emit(end_abs)` 显式收尾），`test_segmenter.py::TestBoundaries` 锁定。
2. **去重比较不同口径**：窗口存去标点核心文本，却拿带标点的 `ja` 去比——
   「ありがとう」vs「ありがとう。」漏判 repeat。已修（同口径比较），
   `test_pipeline.py::TestAsrDecodeDedup::test_punctuation_variant_also_dropped` 锁定。
3. **关机哨兵吞掉最后一条字幕**：`mt_loop` live 分支中 None 紧跟在 live 项后入队时，
   旧实现把该 live 项丢进 late_q 后直接 break——最后一条 live 字幕既不翻译也不上屏。
   已修（先翻译再退），`test_pipeline.py::test_sentinel_right_after_item_still_shows_it` 锁定。

## 已知歧义（测试锁定现状，未改行为）

- **`--mt none` 不上屏**：帮助文本说 "none=ja only"，但 `mt_loop` 的
  `if ... and zh_shown:` 门槛让空译文整条丢弃，字幕窗反而全空；GUI 里的
  `zh or ja` 回落永远等不到空 zh 的 item。是否改成"无翻译时显示日文"待用户决策，
  `test_pipeline.py::test_none_translator_shows_nothing` 锁定现状。
- **回放模式的背压损耗**：replay 全速推流时，`mt_q(maxsize=2)` 会反压 `asr_loop`，
  `seg_q(100)` 满后 `Segmenter` 的 newest-wins 会静默丢段——长音频回放（soak 60m）
  的评测保真度受此影响，需实测 `seg_id` 连续性确认（本次未验证）。
- **feeder 尾部残帧**：读不满一个 chunk 的尾部永远留在 FrameBuffer 里
  （live 路径同样如此），集成测试用补齐到 chunk 整数倍绕过。

## stub 契约

`conftest.py` 里的 stub 只实现生产代码真正触碰的成员——这个成员面本身被
`test_contracts.py::TestMtBackendProtocol` 钉死：任何对象只要有
`translate(text)->str` 和 `last_stats` dict 就能当 MT 后端。
