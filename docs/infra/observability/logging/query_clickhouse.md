# ClickHouse log query helper

**Path:** `infra/setup/clickhouse_query.sh`
**Type:** executable shell script

---

## What it does

* Locates a ClickHouse pod in Kubernetes
* Verifies ClickHouse is ready
* Queries `logs.kube_logs` within a time window
* Filters by `service` (or `namespace` for qdrant)
* Optionally filters by `level`
* Outputs in human or machine-friendly formats
* Supports strict mode for CI

---

## Usage

```text
infra/setup/clickhouse_query.sh \
  --service=<service> \
  [--lastM=<minutes> | --from=<offset> --to=<offset>] \
  [--limit=<N>] \
  [--format=<PrettyCompact|TSV|JSONEachRow>] \
  [--levels=<csv>] \
  [--strict]
```

### Examples

```bash
# Last 10 minutes (default output)
infra/setup/clickhouse_query.sh --service=retrieval --lastM=10

# Last 30 minutes, errors + warnings, JSON output
infra/setup/clickhouse_query.sh \
  --service=frontend \
  --from=30M --to=0M \
  --levels=error,warn \
  --format=JSONEachRow \
  --limit=200

# CI gate: fail if no logs in last 5 minutes
infra/setup/clickhouse_query.sh --service=retrieval --lastM=5 --strict
```

---

## Environment variables

Optional overrides:

* `CH_NS` – ClickHouse namespace (default: `observability`)
* `CH_DB` – database (default: `logs`)
* `CH_TABLE` – table (default: `kube_logs`)

```bash
export CH_NS=observability
export CH_DB=logs
export CH_TABLE=kube_logs
```

---

## Time windows

Choose **one**:

* `--lastM=<minutes>`
  Shortcut for “last N minutes”

* `--from=<offset> --to=<offset>`

Offset format: `<number><unit>`

Units:

* `s` seconds
* `m` minutes
* `h` hours
* `d` days

Examples:

* `30M` → 30 minutes ago
* `0M` → now
* `1H` → 1 hour ago

Rule: `from` must be older than `to`.

---

## Level filtering

`--levels` is **query-time only**.

Accepted tokens:

* `info`
* `warn` / `warning`
* `error` / `err`
* `debug`
* `critical` / `fatal`
* `trace`

Mapped to stored values:

* `INFO`, `WARN`, `ERROR`, `DEBUG`, `CRITICAL`, `TRACE`

Notes:

* Applied only if the `level` column exists
* Unknown tokens are ignored
* This does **not** control ingestion or verbosity
  (that is handled by `LOG_LEVEL` in the application)

---

## Output formats

* `PrettyCompact` (default)
* `TSV`
* `JSONEachRow`

Example:

```bash
infra/setup/clickhouse_query.sh \
  --service=retrieval \
  --lastM=5 \
  --format=JSONEachRow | jq .
```

---

## Service handling

* Normal services: filtered by `service = '<name>'`
* `--service=qdrant`: filtered by `namespace = 'qdrant'`

---

## Schema assumptions

The script adapts to the table but expects:

* Timestamp column (preferred order): `ts`, `_time`, `timestamp`, `time`
* `service` column for service filtering
* `level` column for level filtering (optional)

Missing columns are handled gracefully with warnings.

---

## Exit behavior (CI-safe)

* `0` → success
* Non-zero → error or (with `--strict`) zero matched rows

`--strict` is intended for CI and automation.

---

## Requirements

* `kubectl` configured with access to the cluster
* Permission to `get pods` and `exec` into ClickHouse pod
* ClickHouse pod includes `clickhouse-client`

---

## Operational guidance

* Use `--lastM` for most workflows
* Use `JSONEachRow` for automation
* Use `--strict` in CI
* Keep `SERVICE_NAME` stable in app logs
* Do not rely on query-time filters to reduce log volume

--