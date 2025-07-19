

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_REGION=ap-south-1
export S3_BUCKET_NAME=e2e-rag-system16  # give a complex and unique name
export PYTHONPATH=$(pwd)

export GITHUB_TOKEN=
export GITHUB_USER=
export REPO=RAG8s


flux bootstrap github \
  --owner=$GITHUB_USER \
  --repository=$REPO \
  --branch=main \
  --path=infra/fluxCD/dev \
  --personal

kubectl delete kustomization flux-system -n flux-system
kubectl apply -f infra/fluxCD/flux-system/dev-kustomization.yaml

docker run -d --name sslip-dns --restart=unless-stopped \
  -p 5353:53/udp \
  cunnie/sslip.io-dns-server



export QDRANT_COLLECTION=my_vectors
export ARANGO_DB_NAME=mydb
export ARANGO_USERNAME=root
export ARANGO_PASSWORD=myarrango414

# latest stable as of mid 2025
export QDRANT_IMAGE=qdrant/qdrant:v1.13.4
export VALKEY_IMAGE=valkey/valkey:8.1.3-alpine
export ARANGO_IMAGE=arangodb/arangodb:3.12.5



export CSV_ROWS_PER_CHUNK=



For **multilingual, lightweight entity linking**, here are the most relevant modern replacements to SpEL-base or ReLiK-small:

---

##  Multilingual Lightweight Entity Linking Options

## 📊 Comparison Table

| Model              | Language Coverage                            | Speed Efficiency                     | Accuracy Compared to SpEL-base                                                            | Size / Complexity                          |
| ------------------ | -------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------ |
| **mReFinED**       | 9 languages including low-res ones           | \~44× faster than mGENRE → very fast | Competitive multilingual accuracy; better than prior MEL models ([arXiv][4], [GitHub][2]) | Lightweight, end-to-end                    |
| SpEL-base          | English (monolingual)                        | Moderate (CPU-capable)               | \~92.7% AIDA Test-A (English only)                                                        | \~128 M params                             |

---

* For **multilingual support with a compact, efficient model**, **mReFinED** is the most accurate and CPU-friendly option among modern multilingual EL systems.
* For the **broadest language coverage and ease of use**, **BELA (via MultiEL)** offers lightweight linking across \~98 languages with reliable dual-encoder speed.
* If you only require **English and highest linking accuracy**, stick with **SpEL-base**, but it lacks multilingual capability.


