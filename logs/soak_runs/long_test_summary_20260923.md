# 长时评测总表（2026-09-23，30m/60m × realtime/fast 四轮）

生产配置（与 `start.bat` 一致）：`--model anime --mt sakura --mt-ngl 99 --layer 1
--strategy adaptive --max-s 5.0 --hang-s 2.0`。素材：`benchmarks/soak_30min.wav`
（1800s，sha256:892a5e9c41e7e7ef）与 `benchmarks/soak_60min.wav`（3600s）。
每轮完整条件与逐项统计见同目录 `*.conditions.json` 与 `*.report.md`。

## 四轮对比

| 测试 | 模式 | 墙钟 | ASR 段 | batch_max | 有效翻译 | **显示字幕** | 段丢弃 | 延迟 p50/p90/max |
|---|---|---|---|---|---|---|---|---|
| 30m_realtime | 实时 | 1812s | **158** | 1 | 155 | **50（32%）** | 71 too-short | 4.16/5.73/6.79 |
| 60m_realtime | 实时 | 3612s | **356** | 1 | 349 | **50（14%）** | 117 too-short | 4.06/5.67/6.52 |
| 30m_fast | 全速 | 63.8s | **101** | 100 | 99 | 99（100%） | 71 too-short | 4.54/6.18/7.20 |
| 60m_fast | 全速 | 58.0s | **101** | 100 | 100 | 100（100%） | 117 too-short | 4.17/5.97/6.98 |

## 三个决定性结论

**1. fast 模式的段数被 seg_q 容量锁死（缺陷实锤）**：30m 与 60m 的 fast 跑
**都只产出 101 个 ASR 段**——恰好 = seg_q(100) 容量 + 首批 1 个，与音频长度
完全无关；实时模式则是 158 → 356（音频翻倍段数严格翻倍）。fast 全速推流在
~1 秒内涌出全部段，seg_q 用 `_put_or_bump`（newest-wins），asr_loop 只够
捞走队尾约 100 段，**更早的段在识别前就被驱逐**：60m 音频实际丢失约 72%
的段（356→101），且丢失的是**开场**（幸存段只覆盖音频后段）。
`asr_batch_max=100` 即"一次捞干整队"的证据。

**2. 实时模式的显示数被 result_q 容量锁死（问题八在长时下必发）**：两轮实时
跑的翻译数随音频正常增长（155→349），但**显示数都钉在 50**——恰是
result_q(50) 容量。原因：主线程整个推流期间都在 feed 循环，排水循环要等
推流全部结束才启动，期间 result_q 无消费者，只保住最后约 50 条。
**跑得越久丢得越狠**：30m 丢 68%，60m 丢 86%。

**3. 实时模式下半段以上指标健康可信**：实时两轮 158/356 段全部 `role=live`、
`batch_max=1`（从未积压）、seg_id 0-157/0-355 连续无断档、延迟 p50 ~4.1s
（段长 3.57 + asr 0.32 + mt ~0.2）、段长 p50/p90/max = 3.57/5.1/5.61
（adaptive max-s 5.0 真实生效）、零空译文零拒答。**即：段级数据（asr/mt/
延迟/丢弃统计）实时模式可信；只有 `[done] 显示数`被问题八污染。**

## 对方法论的影响

- **长时测量必须用实时回放**：fast 模式连"处理了多少段"都是错的（容量锁死），
  其唯一价值是量化 seg_q 缺陷（本表第 1 条）。
- **实时回放的显示数不可用**：任何超过约 25 秒 MT 产出的跑，显示数都会被
  锁死在 50。要测显示链路必须先把排水循环改到与推流并行（见 issues 问题八）。
- 语义正确度仍不在长时轨道度量（soak 素材无人工字幕）；准确度看
  `dataset/` 的三个 180s 标准化锚点场景（85.7%~91.7%）。
- 历史 `30m_baseline/30m_adaptive/60m_adaptive` 产物作废（CPU MT + 旧代码 +
  fast 三重问题），不再引用。

## 复现命令

```bash
.venv\Scripts\python.exe tools\run_long_eval.py --name 30m_realtime_20260923 --audio benchmarks\soak_30min.wav
.venv\Scripts\python.exe tools\run_long_eval.py --name 60m_realtime_20260923 --audio benchmarks\soak_60min.wav
.venv\Scripts\python.exe tools\run_long_eval.py --name 30m_fast_20260923     --audio benchmarks\soak_30min.wav --fast
.venv\Scripts\python.exe tools\run_long_eval.py --name 60m_fast_20260923     --audio benchmarks\soak_60min.wav --fast
```
