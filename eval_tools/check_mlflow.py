import mlflow
import pandas as pd

mlflow.set_tracking_uri('sqlite:///D:/ADLINK/asmr-live-sub/mlflow.db')
exp = mlflow.get_experiment_by_name('ASMR-Live-Subtitle-Benchmarks')
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

print(f"Total Runs in Experiment: {len(runs)}")
print("=" * 80)
for _, r in runs.iterrows():
    name = r.get("tags.mlflow.runName", "Unnamed")
    run_id = r["run_id"]
    st = r["start_time"]
    
    # 提取非空核心指标
    metrics = {k.replace("metrics.", ""): round(v, 2) for k, v in r.items() if k.startswith("metrics.") and pd.notna(v)}
    print(f"▶ Run: {name} (ID: {run_id[:8]})")
    print(f"  Time: {st}")
    print(f"  Metrics ({len(metrics)} logged): {metrics}")
    print("-" * 80)
