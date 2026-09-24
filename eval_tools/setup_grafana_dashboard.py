"""Auto-provision Prometheus datasource and custom pipeline dashboard into Grafana."""
import json
import urllib.request
import urllib.error
import base64
import time

GRAFANA_URL = "http://localhost:3000"
AUTH_HEADER = "Basic " + base64.b64encode(b"admin:admin").decode("utf-8")

def api_post(endpoint, payload):
    url = f"{GRAFANA_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Authorization": AUTH_HEADER}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8')
        print(f"[api-warn] {endpoint} -> HTTP {e.code}: {err_msg}")
        return None

def main():
    print("[grafana-setup] Configuring Prometheus data source...")
    ds_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "access": "proxy",
        "url": "http://prometheus:9090",
        "isDefault": True
    }
    res_ds = api_post("/api/datasources", ds_payload)
    if res_ds:
        print("[grafana-setup] Datasource created successfully!")

    print("[grafana-setup] Provisioning ASMR Pipeline Performance Dashboard...")
    dashboard_payload = {
        "dashboard": {
            "id": None,
            "title": "ASMR Live Subtitle - 实时研发效能与质量大屏",
            "tags": ["asmr", "pipeline", "performance"],
            "timezone": "browser",
            "refresh": "5s",
            "schemaVersion": 16,
            "panels": [
                {
                    "title": "端到端全链路物理延迟 (E2E Latency)",
                    "type": "timeseries",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {"expr": "asmr_e2e_latency_seconds", "legendFormat": "端到端总延迟 (s)"},
                        {"expr": "asmr_asr_latency_seconds", "legendFormat": "ASR语音听译耗时 (s)"},
                        {"expr": "asmr_mt_latency_seconds", "legendFormat": "大模型翻译生成耗时 (s)"}
                    ]
                },
                {
                    "title": "端到端当前延迟 (当前实测)",
                    "type": "gauge",
                    "gridPos": {"x": 12, "y": 0, "w": 6, "h": 8},
                    "targets": [{"expr": "asmr_e2e_latency_seconds"}],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "s",
                            "min": 0,
                            "max": 1.0,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 0.35},
                                    {"color": "red", "value": 0.50}
                                ]
                            }
                        }
                    }
                },
                {
                    "title": "显卡显存占用 (VRAM Used)",
                    "type": "gauge",
                    "gridPos": {"x": 18, "y": 0, "w": 6, "h": 8},
                    "targets": [{"expr": "asmr_gpu_vram_used_bytes"}],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "bytes",
                            "min": 0,
                            "max": 8589934592,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 6979321856},
                                    {"color": "red", "value": 7838315520}
                                ]
                            }
                        }
                    }
                },
                {
                    "title": "累计翻译中文字符数 (Total Chars)",
                    "type": "stat",
                    "gridPos": {"x": 0, "y": 8, "w": 12, "h": 4},
                    "targets": [{"expr": "asmr_translation_chars_total"}]
                },
                {
                    "title": "VAD 静音/低质切片丢弃总数 (Drop Count)",
                    "type": "stat",
                    "gridPos": {"x": 12, "y": 8, "w": 12, "h": 4},
                    "targets": [{"expr": "asmr_drop_count_total"}],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "blue", "value": None}
                                ]
                            }
                        }
                    }
                }
            ]
        },
        "overwrite": True
    }
    res_db = api_post("/api/dashboards/db", dashboard_payload)
    if res_db and res_db.get("status") == "success":
        db_url = f"{GRAFANA_URL}{res_db.get('url', '')}"
        print(f"[grafana-setup] Dashboard ready! Access directly at: {db_url}")

if __name__ == "__main__":
    main()
