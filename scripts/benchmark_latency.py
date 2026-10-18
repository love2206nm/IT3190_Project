"""Benchmark Latency, QPS, Memory and CPU usage for UIT-VSMEC Sentiment API."""

import json
import time
import urllib.request
import psutil
import os

ENDPOINT = "http://127.0.0.1:8001/api/v1/predict"
SAMPLES = [
    "Hôm nay trời đẹp và vui quá!",
    "Shop giao hàng chậm, thái độ phục vụ quá kém",
    "Sản phẩm dùng tốt, đóng gói rất kỹ lưỡng ❤️",
    "Tôi cảm thấy rất buồn và cô đơn",
    "Thật bất ngờ khi nhận được món quà này",
]

def benchmark(n_iterations: int = 50):
    latencies = []
    process = psutil.Process(os.getpid())
    
    # Warmup
    for text in SAMPLES:
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            resp.read()

    # Đo đạc
    t_start = time.time()
    for i in range(n_iterations):
        text = SAMPLES[i % len(SAMPLES)]
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            resp.read()
        latencies.append((time.perf_counter() - t0) * 1000)

    total_time = time.time() - t_start
    qps = n_iterations / total_time
    
    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p50 = latencies[int(len(latencies) * 0.50)]
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    min_lat = min(latencies)
    max_lat = max(latencies)

    # Tìm process Uvicorn để đo RAM
    uvicorn_mem = 0.0
    uvicorn_cpu = 0.0
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "uvicorn" in cmd and "8001" in cmd:
                uvicorn_mem = p.memory_info().rss / (1024 * 1024)
                uvicorn_cpu = p.cpu_percent(interval=0.1)
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    report = {
        "n_requests": n_iterations,
        "total_time_sec": round(total_time, 3),
        "qps": round(qps, 2),
        "latency_ms": {
            "min": round(min_lat, 2),
            "avg": round(avg_latency, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max_lat, 2),
        },
        "server_resources": {
            "uvicorn_ram_mb": round(uvicorn_mem, 2),
            "uvicorn_cpu_pct": round(uvicorn_cpu, 2),
        }
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    benchmark(100)
