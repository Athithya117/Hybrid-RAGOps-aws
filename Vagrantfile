Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"
  config.vm.hostname = "rag8s"
  config.ssh.insert_key = false

  # Synced folder
  config.vm.synced_folder ".", "/vagrant", type: "rsync"

  # Private network IP
  config.vm.network "private_network", ip: "192.168.56.18"

  # === Port Forwarding ===
  # Load Balancer
  config.vm.network "forwarded_port", guest: 80, host: 80, auto_correct: true     # HTTP
  config.vm.network "forwarded_port", guest: 443, host: 443, auto_correct: true   # HTTPS

  # Registry
  config.vm.network "forwarded_port", guest: 5000, host: 5000, auto_correct: true # Docker registry

  # Observability
  config.vm.network "forwarded_port", guest: 9090, host: 9090, auto_correct: true # Prometheus
  config.vm.network "forwarded_port", guest: 3000, host: 3000, auto_correct: true # Grafana
  config.vm.network "forwarded_port", guest: 9093, host: 9093, auto_correct: true # Alertmanager

  # Ray
  config.vm.network "forwarded_port", guest: 8000, host: 8000, auto_correct: true # Ray Serve / RAG UI
  config.vm.network "forwarded_port", guest: 9000, host: 9000, auto_correct: true # Ray gRPC
  config.vm.network "forwarded_port", guest: 8265, host: 8265, auto_correct: true # Ray dashboard

  # Qdrant
  config.vm.network "forwarded_port", guest: 6333, host: 6333, auto_correct: true # Qdrant HTTP
  config.vm.network "forwarded_port", guest: 6334, host: 6334, auto_correct: true # Qdrant gRPC

  # ArangoDB
  config.vm.network "forwarded_port", guest: 8529, host: 8529, auto_correct: true # ArangoDB Web UI

  # Keycloak
  config.vm.network "forwarded_port", guest: 8443, host: 8443, auto_correct: true # Keycloak Auth (HTTPS)

  # Postgres
  config.vm.network "forwarded_port", guest: 5432, host: 5432, auto_correct: true # PostgreSQL

  # RAGApp frontend (optional)
  config.vm.network "forwarded_port", guest: 3001, host: 3001, auto_correct: true # React frontend

  # Model inference ports
  config.vm.network "forwarded_port", guest: 8501, host: 8501, auto_correct: true
  config.vm.network "forwarded_port", guest: 8502, host: 8502, auto_correct: true
  config.vm.network "forwarded_port", guest: 8503, host: 8503, auto_correct: true
  config.vm.network "forwarded_port", guest: 8504, host: 8504, auto_correct: true
  config.vm.network "forwarded_port", guest: 8505, host: 8505, auto_correct: true

  # VM resources
  config.vm.provider "virtualbox" do |vb|
    vb.memory = 11000
    vb.cpus = 6
    vb.gui = false
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
  end

  # Provisioning
  config.vm.provision "shell", privileged: true, inline: <<-SHELL
    set -eux

    # === Fix DNS resolution (systemd-resolved workaround) ===
    echo "nameserver 1.1.1.1" > /etc/resolv.conf
    echo "nameserver 8.8.8.8" >> /etc/resolv.conf

    # === Base tools ===
    apt-get update -y
    apt-get install -y make curl git gh tree ca-certificates gnupg lsb-release software-properties-common unzip build-essential

    # === Docker Installation ===
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list

    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    groupadd docker || true
    usermod -aG docker vagrant
    systemctl enable docker
    systemctl start docker
  SHELL
end
