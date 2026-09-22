# 多参考译文对比

- live: `logs/benchmark_runs/custom_random_talk.jsonl`（剔除联动丢 5 段）
- gt: `dataset/scene_random_talk.gt_ja.json`

## 实时 vs 各参考（端到端差距）

| 参考 | 相似度 |
| :--- | :--- |
| sakura | 71.08% |
| stepflash | 56.77% |

## 参考层内部（相互距离；人工译文是锚点）

| A \ B | sakura | stepflash |
| :--- | :--- | :--- |
| sakura | — | 56.64% |
| stepflash | 56.64% | — |
