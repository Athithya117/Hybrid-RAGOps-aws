

user@LAPTOP-JM3HED0O:~/RAG8s$ tree
.
├── Makefile
├── README.md
├── apps
│   ├── dense
│   │   ├── Dockerfile
│   │   └── host_dense.py
│   ├── index
│   │   ├── Dockerfile
│   │   ├── index.py
│   │   ├── indexing_pipeline.py
│   │   ├── models
│   │   │   └── faster_whisper
│   │   │       └── faster-whisper-base
│   │   │           ├── README.md
│   │   │           ├── config.json
│   │   │           ├── model.bin
│   │   │           ├── tokenizer.json
│   │   │           └── vocabulary.txt
│   │   ├── parse_chunk
│   │   │   ├── __init__.py
│   │   │   ├── formats
│   │   │   │   ├── __init__.py
│   │   │   │   ├── _csv.py
│   │   │   │   ├── html.py
│   │   │   │   ├── images.py
│   │   │   │   ├── jsonl.py
│   │   │   │   ├── md.py
│   │   │   │   ├── pdf.py
│   │   │   │   ├── pptx.py
│   │   │   │   ├── txt.py
│   │   │   │   └── wav.py
│   │   │   └── router.py
│   │   ├── pre_conversions
│   │   │   ├── Dockerfile
│   │   │   └── convert_all.sh
│   │   └── requirements.txt
│   ├── inference
│   │   ├── eval
│   │   │   └── eval.py
│   │   ├── frontend
│   │   │   ├── Dockerfile
│   │   │   ├── frontend_ui.py
│   │   │   └── requirements.txt
│   │   └── retrieval
│   │       ├── Dockerfile
│   │       ├── query.py
│   │       └── requirements.txt
│   ├── reranker
│   │   ├── Dockerfile
│   │   └── host_reranker.py
│   └── sparse
│       ├── Dockerfile
│       └── host_sparse.py
├── docs
│   ├── README.md
│   ├── architecture
│   │   ├── README.md
│   │   ├── component_ownership.md
│   │   ├── data_flow.md
│   │   └── high_level_diagram.md
│   ├── infra
│   │   ├── networking
│   │   │   ├── README.md
│   │   │   ├── alb.md
│   │   │   ├── ingress.md
│   │   │   └── vpc.md
│   │   ├── pulumi
│   │   │   ├── README.md
│   │   │   ├── aws_architecture.md
│   │   │   ├── eks.md
│   │   │   └── iam.md
│   │   └── security
│   │       ├── README.md
│   │       ├── oidc.md
│   │       ├── policies.md
│   │       └── rbac.md
│   ├── operations
│   │   ├── alerts
│   │   │   ├── README.md
│   │   │   ├── alert_catalog.md
│   │   │   └── alert_runbooks.md
│   │   ├── backup
│   │   │   ├── README.md
│   │   │   ├── qdrant_backup.md
│   │   │   └── s3_backup.md
│   │   ├── deployments
│   │   │   ├── README.md
│   │   │   ├── release_flow.md
│   │   │   └── rollback.md
│   │   └── monitoring
│   │       ├── README.md
│   │       ├── logs.md
│   │       ├── metrics.md
│   │       └── prometheus_rules.md
│   ├── overview
│   │   ├── README.md
│   │   ├── glossary.md
│   │   └── intro.md
│   ├── platform
│   │   ├── argocd
│   │   │   ├── README.md
│   │   │   ├── applicationset.md
│   │   │   ├── root_app.md
│   │   │   └── sync_policies.md
│   │   ├── generators
│   │   │   ├── README.md
│   │   │   ├── generator_design.md
│   │   │   └── templates.md
│   │   └── manifests
│   │       ├── README.md
│   │       ├── k8s_core.md
│   │       ├── karpenter.md
│   │       ├── qdrant.md
│   │       └── rag.md
│   ├── rag
│   │   ├── indexing
│   │   │   ├── README.md
│   │   │   ├── chunking_formats.md
│   │   │   └── pipeline.md
│   │   ├── inference
│   │   │   ├── README.md
│   │   │   ├── frontend.md
│   │   │   └── query_service.md
│   │   └── models
│   │       ├── README.md
│   │       ├── dense.md
│   │       ├── reranker.md
│   │       └── sparse.md
│   ├── references
│   │   ├── README.md
│   │   ├── api_specs.md
│   │   └── links.md
│   ├── runbooks
│   │   ├── README.md
│   │   ├── incident_response.md
│   │   └── oncall_checklist.md
│   └── troubleshooting
│       ├── README.md
│       ├── common_issues.md
│       ├── eks.md
│       └── networking.md
├── infra
│   ├── generators
│   │   ├── gen_ingress.py
│   │   ├── gen_k8s_core.py
│   │   ├── gen_karpenter.py
│   │   ├── gen_monitoring.py
│   │   ├── gen_qdrant.py
│   │   ├── gen_rag_services.py
│   │   ├── generator_applicationset.py
│   │   └── templates
│   │       ├── deployment.j2
│   │       ├── ingress.j2
│   │       ├── karpenter-provisioner.j2
│   │       ├── namespace.j2
│   │       ├── prometheus-rules.j2
│   │       ├── service.j2
│   │       └── statefulset.j2
│   ├── manifests
│   │   ├── argocd
│   │   │   ├── applicationset.yaml
│   │   │   └── root-app.yaml
│   │   ├── ingress
│   │   ├── k8s-core
│   │   ├── karpenter
│   │   ├── monitoring
│   │   ├── qdrant
│   │   └── rag
│   └── pulumi_aws
│       ├── __main__.py
│       ├── acm.py
│       ├── alb.py
│       ├── cognito.py
│       ├── config.py
│       ├── eks_cluster.py
│       ├── karpenter_setup.py
│       ├── outputs.py
│       ├── route53.py
│       ├── utils
│       │   ├── policies.py
│       │   ├── tags.py
│       │   └── validators.py
│       └── vpc.py
└── scripts
    ├── bootstrap_dev.sh
    ├── bootstrap_prod.sh
    ├── lc.sh
    ├── s3_buckets.py
    └── sync_data_with_s3.py

53 directories, 137 files
user@LAPTOP-JM3HED0O:~/RAG8s$ 