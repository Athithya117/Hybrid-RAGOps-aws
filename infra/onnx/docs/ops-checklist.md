Pre-deploy checklist (production)

- CI: all builds green & scanner reports no critical CVEs.
- Smoke tests: tests/smoke_test.sh passed in staging.
- Secrets: HF token present in cluster (validate: kubectl -n <ns> get secret <name>).
- Capacity: node autoscaler enabled and tested with load.
- Alerts: Prometheus alerts configured for high latency / high error rate.
- Backup: any important persistent state backed up.
- Rollback plan: have previous image tag & helm command ready.
