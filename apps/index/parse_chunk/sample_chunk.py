import os, io, json
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

conn=os.environ["AZURE_STORAGE_CONNECTION_STRING"]
container=os.environ["AZURE_CONTAINER"]
key="data/chunked/864c4606b2f8f74e8c25d8fef3b4a5f6623e3ba2798d8037640629be0afed834.parquet"

svc=BlobServiceClient.from_connection_string(conn)
blob=svc.get_container_client(container).get_blob_client(key)
buf=io.BytesIO(blob.download_blob().readall())

table=pq.read_table(buf)
rows=table.to_pylist()

print(json.dumps(rows,indent=2,ensure_ascii=False))
