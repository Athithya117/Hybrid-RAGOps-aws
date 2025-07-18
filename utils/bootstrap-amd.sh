

sudo apt install -y unzip curl
pip install --upgrade pip wheel
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install && rm -rf awscliv2.zip aws

mkdir -p backups/dbs/qdrant
mkdir -p backups/dbs/arrangodb
mkdir -p backups/dbs/valkeye
