#!/usr/bin/env python3
"""
Deterministic test runner for alerting pipeline.

- Starts kubectl port-forwards for victoria-metrics, vmalert and alertmanager
- Pushes deterministic synthetic metrics
- Posts four synthetic alerts:
  * two paging (critical) alerts (qdrant-paging, retriever-paging)
  * two non-paging (warning) alerts (test-channel1-nonpaging, test-channel2-nonpaging)
- Verifies Alertmanager shows the alerts and that receiver chosen matches PLATFORM env config:
  ENABLE_PAGERDUTY, ENABLE_SLACK, ALERTING_PAGING_SEVERITY_LEVELS, ALERTING_SLACK_SEVERITY_LEVELS
- Verifies Victoria ingestion of the test series
"""

from __future__ import annotations
import os
import sys
import time
import uuid
import socket
import subprocess
import signal
import json
from pathlib import Path
from typing import List, Dict
import urllib.request
import urllib.parse

NS = os.getenv("NS", "monitoring")
VICTORIA_SVC = os.getenv("VICTORIA_SVC", "victoria-metrics")
VMALERT_SVC = os.getenv("VMALERT_SVC", "vmalert")
ALERTM_SVC = os.getenv("ALERTM_SVC", "alertmanager")

ENABLE_PAGERDUTY = os.getenv("ENABLE_PAGERDUTY", "true").lower() in ("1", "true", "yes", "on")
ENABLE_SLACK = os.getenv("ENABLE_SLACK", "true").lower() in ("1", "true", "yes", "on")
ALERTING_PAGING_SEVERITY_LEVELS = os.getenv("ALERTING_PAGING_SEVERITY_LEVELS", "critical")
ALERTING_SLACK_SEVERITY_LEVELS = os.getenv("ALERTING_SLACK_SEVERITY_LEVELS", "warning,critical")

TMPDIR = Path("/tmp/test_alerting") / uuid.uuid4().hex
TMPDIR.mkdir(parents=True, exist_ok=True)
LOGFILE_PREFIX = TMPDIR / "pf"
PIDS: List[subprocess.Popen] = []

def find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p

def start_portforward(svc: str, local_port: int, target_port: int, logfile: Path) -> subprocess.Popen:
    cmd = ["kubectl", "-n", NS, "port-forward", f"svc/{svc}", f"{local_port}:{target_port}"]
    fh = open(str(logfile), "w")
    p = subprocess.Popen(cmd, stdout=fh, stderr=fh, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    PIDS.append(p)
    return p

def wait_http(url: str, timeout: int = 20) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status < 400:
                    return True
        except Exception:
            time.sleep(1)
    return False

def push_metrics(victoria_local: int, test_id: str) -> None:
    now_ms = int(time.time() * 1000)
    t1 = now_ms - 120000
    t2 = now_ms - 60000
    metrics = (
        f'retrieval_requests_total{{service="retrieval",test_run="{test_id}"}} 100 {t1}\n'
        f'retrieval_errors_total{{service="retrieval",test_run="{test_id}"}} 60 {t1}\n'
        f'retrieval_requests_total{{service="retrieval",test_run="{test_id}"}} 200 {t2}\n'
        f'retrieval_errors_total{{service="retrieval",test_run="{test_id}"}} 140 {t2}\n'
        f'retrieval_requests_total{{service="retrieval",test_run="{test_id}"}} 300 {now_ms}\n'
        f'retrieval_errors_total{{service="retrieval",test_run="{test_id}"}} 220 {now_ms}\n'
    )
    url = f"http://127.0.0.1:{victoria_local}/api/v1/import/prometheus"
    req = urllib.request.Request(url, data=metrics.encode("utf-8"), method="POST", headers={"Content-Type":"application/octet-stream"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        # benign: Victoria may respond with JSON; we do not assert content here

def post_alert(alertmanager_local: int, labels: Dict[str,str], annotations: Dict[str,str]) -> None:
    payload = [{"labels": labels, "annotations": annotations}]
    data = json.dumps(payload).encode("utf-8")
    url = f"http://127.0.0.1:{alertmanager_local}/api/v2/alerts"
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            pass
    except Exception:
        # Alertmanager may accept but return no body; ignore network errs for now
        pass

def get_alerts(alertmanager_local: int) -> List[Dict]:
    url = f"http://127.0.0.1:{alertmanager_local}/api/v2/alerts"
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except Exception:
        return []

def query_victoria(victoria_local: int, query: str) -> Dict:
    url = f"http://127.0.0.1:{victoria_local}/api/v1/query?{urllib.parse.urlencode({'query': query})}"
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return {}

def determine_expected_receiver(plane: str, severity: str) -> str:
    sev = severity.lower()
    paging = [s.strip().lower() for s in ALERTING_PAGING_SEVERITY_LEVELS.split(",") if s.strip()]
    slack = [s.strip().lower() for s in ALERTING_SLACK_SEVERITY_LEVELS.split(",") if s.strip()]
    if sev in paging and ENABLE_PAGERDUTY:
        return "pagerduty"
    if sev in slack and ENABLE_SLACK:
        return "slack"
    if sev in paging and not ENABLE_PAGERDUTY and ENABLE_SLACK:
        return "slack"
    return "default"

def cleanup():
    for p in PIDS:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(0.5)
    for p in PIDS:
        try:
            p.kill()
        except Exception:
            pass

def main():
    test_id = str(uuid.uuid4())
    victoria_local = find_free_port()
    vmalert_local = find_free_port()
    alertm_local = find_free_port()
    vic_log = TMPDIR / "victoria.log"
    vmalert_log = TMPDIR / "vmalert.log"
    alertm_log = TMPDIR / "alertm.log"
    print(f"INFO starting port-forwards victoria={victoria_local} vmalert={vmalert_local} alertm={alertm_local}")
    start_portforward(VICTORIA_SVC, victoria_local, 8428, vic_log)
    start_portforward(VMALERT_SVC, vmalert_local, 8080, vmalert_log)
    start_portforward(ALERTM_SVC, alertm_local, 9093, alertm_log)
    try:
        if not wait_http(f"http://127.0.0.1:{alertm_local}/api/v2/status", timeout=20):
            print("ERROR Alertmanager not responding; check logs:", alertm_log)
            cleanup()
            sys.exit(2)
        if not wait_http(f"http://127.0.0.1:{vmalert_local}/metrics", timeout=20):
            print("ERROR vmalert not responding; check logs:", vmalert_log)
            cleanup()
            sys.exit(3)
        print("INFO pushing synthetic metrics")
        push_metrics(victoria_local, test_id)
        time.sleep(1)
        # Post a sanity alert to warm paths
        post_alert(alertm_local, {"alertname":"SanityWarm","severity":"info","plane":"slo","service":"sanity","test_run":test_id}, {"summary":"sanity warm"})
        # Define alerts: two paging, two non-paging
        alerts = [
            ("qdrant-paging", "safety", "critical", "qdrant"),
            ("retriever-paging", "safety", "critical", "retriever"),
            ("test-channel1-nonpaging", "slo", "warning", "test-channel1"),
            ("test-channel2-nonpaging", "slo", "warning", "test-channel2"),
        ]
        for name, plane, sev, svc in alerts:
            post_alert(alertm_local,
                       {"alertname": name, "plane": plane, "severity": sev, "service": svc, "test_run": test_id},
                       {"summary": f"synthetic {name}"})
        # Give Alertmanager a short moment to evaluate routes
        time.sleep(4)
        am_alerts = get_alerts(alertm_local)
        # verify each alert exists and receiver matches expected
        success = True
        for name, plane, sev, svc in alerts:
            matches = [a for a in am_alerts if a.get("labels", {}).get("alertname") == name and a.get("labels", {}).get("service") == svc]
            if not matches:
                print(f"ERROR alert {name} not found in Alertmanager")
                success = False
                continue
            got_receiver = (matches[0].get("receivers") or [{"name": ""}])[0].get("name", "")
            expected = determine_expected_receiver(plane, sev)
            print(f"INFO found alert {name} severity={sev} plane={plane} receiver={got_receiver} expected={expected}")
            if expected != "default" and got_receiver != expected:
                print(f"ERROR unexpected receiver for {name}: got={got_receiver} want={expected}")
                success = False
        # Verify Victoria ingestion
        qres = query_victoria(victoria_local, f'retrieval_requests_total{{test_run="{test_id}"}}')
        if not qres.get("data", {}).get("result"):
            print("ERROR Victoria did not show ingested series for test_run")
            success = False
        else:
            print("INFO Victoria ingestion verified (compact)")
        if not success:
            print("TESTS FAILED")
            cleanup()
            sys.exit(4)
        print("ALL CHECKS PASSED")
    finally:
        cleanup()
        # keep logs for inspection but do not remove automatically
        print(f"Logs kept in {TMPDIR}")

if __name__ == "__main__":
    main()
