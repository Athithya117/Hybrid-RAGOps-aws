

# Create directory and set permissions for sglang
RUN mkdir -p /opt/sglang/cache && chmod -R 777 /opt/sglang

# SGLANG-specific environment variables
ENV SGLANG_HOME=/opt/sglang \
    SGLANG_CACHE=/opt/sglang/cache


    
# Ensure /opt/sglang exists and is writable
sudo mkdir -p /opt/sglang
sudo chmod -R 777 /opt/sglang

# Add environment variables to ~/.bashrc if not already present
grep -qxF 'export SGLANG_HOME="/opt/sglang"' ~/.bashrc || echo 'export SGLANG_HOME="/opt/sglang"' >> ~/.bashrc
grep -qxF 'export SGLANG_CACHE="$SGLANG_HOME/cache"' ~/.bashrc || echo 'export SGLANG_CACHE="$SGLANG_HOME/cache"' >> ~/.bashrc
