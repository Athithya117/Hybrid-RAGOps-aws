






/ (repo root)
├── apps/                                   # App sources (owner: app engineers). Not infra-managed directly.
│   ├── dense/
│   │   ├── Dockerfile                      # Build image for dense embedder; owner: app; used by CI image pipeline. (Coupling: Low)
│   │   ├── host_dense.py                   # Dense model HTTP entrypoint; reads MODEL_* envs at runtime. (Coupling: Low)
│   │   ├── test_and_push_dense.sh          # Local CI helper to build/push image; not used by Pulumi. (Coupling: Low)
│   │   └── test_dense_locally.sh           # Local dev runner script. (Coupling: Low)
│   ├── index/
│   │   ├── build_and_push_image.sh         # Build+push helper for indexer image. (Coupling: Low)
│   │   ├── Dockerfile                      # Indexer container build. (Coupling: Low)
│   │   ├── indexing_pipeline.py            # Orchestrates ingestion -> chunking -> embeddings -> upsert; used by indexer Deployment. (Coupling: Med)
│   │   ├── index.py                        # Idempotent indexer logic; writes to blob & vector DB. (Coupling: High to qdrant/storage)
│   │   ├── parse_chunk/                    # Parsers for many formats; library used by indexer. (Coupling: High for ingestion)
│   │   │   ├── formats/                    # Per-format extractors (owner: app). (Coupling: Low)
│   │   │   │   ├── _csv.py                 # CSV parsing rules. (Coupling: Low)
│   │   │   │   ├── _html.py                # HTML extractor / sanitizer. (Coupling: Low)
│   │   │   │   ├── images.py               # OCR glue; may call external OCR service. (Coupling: Low)
│   │   │   │   ├── __init__.py             # Parser registry. (Coupling: Low)
│   │   │   │   ├── jsonl.py                # JSONL parser. (Coupling: Low)
│   │   │   │   ├── md.py                   # Markdown parser. (Coupling: Low)
│   │   │   │   ├── pdf.py                  # PDF extraction wrapper. (Coupling: Low)
│   │   │   │   ├── _pptx.py                # PPTX extractor. (Coupling: Low)
│   │   │   │   ├── txt.py                  # Plain text normalization. (Coupling: Low)
│   │   │   │   └── wav.py                  # Audio chunking preproc. (Coupling: Low)
│   │   │   ├── __init__.py                 # Parser package init. (Coupling: Low)
│   │   │   └── router.py                   # Dispatch parser per MIME/extension. (Coupling: Low)
│   │   ├── pre_conversions.py              # Batch conversion helpers for CI. (Coupling: Low)
│   │   └── requirements.txt                # Indexer pinned deps. (Coupling: Low)
│   ├── inference/
│   │   ├── eval/
│   │   │   └── eval.py                     # Offline evaluation harness. (Coupling: Low)
│   │   ├── frontend/
│   │   │   ├── Dockerfile                  # Frontend image build. (Coupling: Low)
│   │   │   ├── frontend_and_auth.py        # UI server + OIDC client; reads OIDC envs (OIDC_ISSUER, SPA_CLIENT_ID). (Coupling: High to auth)
│   │   │   └── requirements.txt            # Frontend deps. (Coupling: Low)
│   │   └── retrieval/
│   │       ├── Dockerfile                  # Retrieval service image. (Coupling: Low)
│   │       ├── query_helpers.py            # Helpers used by query.py. (Coupling: Low)
│   │       ├── query.py                    # Query endpoint; validates JWTs & calls model hosts / Qdrant. (Coupling: High)
│   │       └── requirements.txt            # Retrieval deps. (Coupling: Low)
│   ├── reranker/
│   │   ├── Dockerfile                      # Reranker image. (Coupling: Low)
│   │   ├── host_reranker.py                # Reranker server entrypoint; reads model path envs. (Coupling: Low)
│   │   ├── test_and_push_reranker.sh      # CI helper. (Coupling: Low)
│   │   └── test_reranker_locally.sh       # Local dev runner. (Coupling: Low)
│   └── sparse/
│       ├── Dockerfile                      # Sparse retrieval image. (Coupling: Low)
│       └── host_sparse.py                  # Sparse host process. (Coupling: Low)
├── data/
│   └── samples_for_preconversions/         # Sample data used by preconversion tests. (Coupling: Low)
│       ├── random_news.mp3
│       ├── sample-files.com-basic-text.docx
│       └── Supermarket-Sales-Sample-Data.xlsx
├── .devcontainer/                          # VSCode devcontainer config (owner: dev environment). (Coupling: Low)
│   ├── devcontainer.json
│   ├── Dockerfile
│   └── setup.png
├── .dockerignore                           # Docker ignore used by builds. (Coupling: Low)
├── .gitignore                              # Git ignore. (Coupling: Low)
├── infra/
│   ├── base_infra/
│   │   ├── force_sync_azure_and_local_fs.py # CLI helper: sync blob container contents to local FS; uses AZURE_* envs. (Owner: infra; Coupling: Med)
│   │   ├── get_storage_conn_string.py        # Retrieve storage connection string (az CLI or SDK); used by dev scripts. (Coupling: Med)
│   │   └── storage_acc_uai.py               # CLI helper to create storage UAI & assign RBAC; used if Pulumi not owning UAI. (Coupling: Med)
│   ├── generators/                          # Generators emit Kubernetes manifests consumed by Flux (owner: infra)
│   │   ├── cloudflared.py                   # Generates cloudflared manifests; SECRET-safe mode; writes k8s manifests and optionally applies secret via kubectl. (Coupling: High to edge)
│   │   ├── dense.py                         # Generates manifests for dense service (deploy/svc) and inputs-hash. (Coupling: Med)
│   │   ├── frontend_alerts.yaml             # Static alert rules for frontend (already a manifest). (Coupling: Low)
│   │   ├── frontend_auth.py                 # Generates frontend auth config (consumes oidc outputs). (Coupling: High to auth)
│   │   ├── indexing_cronjob.py              # Generates CronJob manifest for indexer. (Coupling: Med)
│   │   ├── monitoring.py                    # Generates Prometheus/Grafana related manifests. (Coupling: Med)
│   │   ├── qdrant_cluster.py                # Generates qdrant StatefulSet, PDBs, Service, backup CronJob. (Coupling: High to storage/aks)
│   │   ├── reranker.py                      # Generates reranker k8s manifests. (Coupling: Med)
│   │   ├── retriever.py                     # Generates retrieval API manifests (deploy/svc). (Coupling: High)
│   │   └── sparse.py                        # Generates sparse host manifests. (Coupling: Med)
│   ├── helm-values/                         # Prefilled values for Helm charts (owner: infra)
│   │   ├── local/
│   │   │   └── values.yaml                  # Local overrides for Helm charts. (Coupling: Low)
│   │   └── qdrant/
│   │       └── values.yaml                  # Qdrant-specific Helm values (PVC sizes, mem). (Coupling: High to qdrant generator)
│   ├── manifests/                           # Generated (checked-in) manifests (owner: generated artifacts)
│   │   ├── dense/
│   │   │   ├── 00-namespace.yaml
│   │   │   ├── 01-sa-role.yaml
│   │   │   ├── 02-deployment.yaml
│   │   │   ├── 03-service.yaml
│   │   │   ├── .inputs_hash
│   │   │   └── last_deploy_summary.json    # Generated deployment artifacts; do not edit; Flux watches repo. (Coupling: High)
│   │   ├── frontend/
│   │   │   ├── 00-namespace.yaml
│   │   │   ├── 01-sa-role.yaml
│   │   │   ├── 02-configmap.yaml
│   │   │   ├── 04-deployment.yaml
│   │   │   ├── 05-service.yaml
│   │   │   └── .inputs_hash
│   │   ├── jobs/
│   │   │   ├── 00-namespace.yaml
│   │   │   ├── 10-serviceaccount.yaml
│   │   │   ├── 20-role.yaml
│   │   │   ├── 30-rolebinding.yaml
│   │   │   ├── 40-secret-azure-placeholder.yaml # Placeholder secret manifests; replaced via ExternalSecret or generator. (Coupling: High)
│   │   │   ├── 41-secret-qdrant-placeholder.yaml
│   │   │   └── 50-cronjob.yaml
│   │   ├── reranker/
│   │   │   ├── 00-namespace.yaml
│   │   │   ├── 01-sa-role.yaml
│   │   │   ├── 02-deployment.yaml
│   │   │   ├── 03-service.yaml
│   │   │   ├── .inputs_hash
│   │   │   └── last_deploy_summary.json
│   │   └── sparse/
│   │       ├── 00-namespace.yaml
│   │       ├── 01-sa-role.yaml
│   │       ├── 02-deployment.yaml
│   │       ├── 03-service.yaml
│   │       ├── .inputs_hash
│   │       └── last_deploy_summary.json
│   ├── pulumi_azure/                        # Pulumi program for Azure (owner: infra, Pulumi)
│   │   ├── aks.py                           # Creates AKS and agent pools; exports kubeconfig + cluster name. (Coupling: High)
│   │   ├── auth.py                          # Creates/imports App registrations; exports oidc_issuer, spa/api client IDs/secrets. (Coupling: High)
│   │   ├── bootstrap.sh                     # CLI bootstrap for Pulumi backend resources (az CLI one-time). (Coupling: Low)
│   │   ├── core_network.py                  # VNet, subnets, storage account, blob container; exports resource ids. (Coupling: High)
│   │   ├── edge.py                          # Decision component; exports frontend_public_url and edge_mode (Pulumi-only, no CF resources). (Coupling: High)
│   │   ├── __main__.py                      # Pulumi stack entrypoint; composes modules and must export outputs used by generators. (Coupling: Critical)
│   │   ├── pulumi-exports.sh                # Shell helper to source pulumi outputs (convenience). (Coupling: Low)
│   │   ├── pulumi-outputs.json              # Pulumi outputs JSON (consumed by generators) — canonical contract. (Coupling: Critical)
│   │   ├── .pulumi_preview.err              # Pulumi preview error snapshot (operational). (Coupling: Low)
│   │   ├── .pulumi_preview.json             # Pulumi preview JSON (operational). (Coupling: Low)
│   │   ├── Pulumi.staging.yaml              # Stack config for staging (pinned config keys read by Pulumi modules). (Coupling: High)
│   │   ├── Pulumi.yaml                      # Pulumi project metadata. (Coupling: Low)
│   │   ├── requirements.txt                  # Python deps for Pulumi program (pulumi/pulumi_azure_native/etc). (Coupling: Low)
│   │   ├── run.sh                            # Wrapper to run pulumi up + export outputs; uses PULUMI_* envs. (Coupling: Low)
│   │   ├── storage_account.py                # CLI helper to create storage account + containers; used pre-Pulumi or to import existing. (Coupling: Med)
│   │   └── uai_key_vault_secrets.py          # CLI helper to create UAIs + Key Vault + seed secrets; used pre-Pulumi. (Coupling: Med)
│   ├── runners/                              # Operational runners (owner: infra ops)
│   │   ├── local_test.sh                     # Quick local test runner. (Coupling: Low)
│   │   ├── run_indexing_cronjob.py           # Runner to execute indexer locally for test/repair. (Coupling: Low)
│   │   ├── run_qdrant_backup.py              # Scripting for backup orchestration. (Coupling: High to qdrant)
│   │   └── run_qdrant_restore.py             # Restore orchestration helper. (Coupling: High to qdrant)
│   ├── scripts/
│   │   └── setup_fluxcd.py                   # Bootstraps Flux into cluster and creates GitRepository/Kustomizations; uses pulumi outputs. (Coupling: High)
│   └── tests/                                # Infra tests (owner: infra)
│       ├── hybrid_support_test.sh            # Hybrid infra capability tests. (Coupling: Low)
│       ├── qdrant_full_restore.py            # End-to-end restore integration test script. (Coupling: High)
│       ├── test_models.sh                    # Model tests (app-level). (Coupling: Low)
│       ├── test_qdrant_full_cluster_backup_restore.py # Integration test for qdrant backup/restore. (Coupling: High)
│       ├── test_qdrant.sh                    # Qdrant smoke tests. (Coupling: Low)
│       └── validate_restore.py               # Validate restore artifacts. (Coupling: Low)
├── Makefile                                  # Developer commands (build/generate/test). (Coupling: Low)
├── README.md                                 # Repo onboarding and architecture summary. (Coupling: Low)
└── utils/
    ├── bootstrap_full.sh                      # Orchestrates pulumi + generators + git commit (CI). (Coupling: High)
    ├── bootstrap.sh                           # Minimal bootstrap helper for dev. (Coupling: Low)
    ├── fix_kind_cluster_dns.sh                # Helper for kind DNS adjustments. (Coupling: Low)
    └── s3_buckets.py                          # Old AWS S3 helper (deprecated; remove or port to Azure Blob). (Coupling: Low)
