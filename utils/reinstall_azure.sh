pip uninstall -y azure-core azure-identity azure-mgmt-storage azure-mgmt-core adlfs || true

pip install --no-cache-dir \
  azure-core==1.30.2 \
  azure-identity==1.16.0 \
  azure-mgmt-core==1.4.0 \
  azure-mgmt-storage==21.2.1
