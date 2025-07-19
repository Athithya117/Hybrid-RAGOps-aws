Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"
  config.vm.hostname = "rag8s"
  config.ssh.insert_key = false

  # === VM Resources (memory and CPU) ===
  config.vm.provider "virtualbox" do |vb|
    vb.memory = 11000
    vb.cpus = 6
    vb.gui = false
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
  end

  # === Synced folder (disabled) ===
  config.vm.synced_folder ".", "/vagrant", disabled: true

  # === Private Network IP ===
  config.vm.network "private_network", ip: "192.168.56.18"

  # === Port Forwarding ===
  # Load Balancer
  config.vm.network "forwarded_port", guest: 80, host: 80, auto_correct: true
  config.vm.network "forwarded_port", guest: 443, host: 443, auto_correct: true

  # Registry
  config.vm.network "forwarded_port", guest: 5000, host: 5000, auto_correct: true

  # Observability
  config.vm.network "forwarded_port", guest: 9090, host: 9090, auto_correct: true # Prometheus
  config.vm.network "forwarded_port", guest: 3000, host: 3000, auto_correct: true # Grafana
  config.vm.network "forwarded_port", guest: 9093, host: 9093, auto_correct: true # Alertmanager

  # Ray
  config.vm.network "forwarded_port", guest: 8000, host: 8000, auto_correct: true
  config.vm.network "forwarded_port", guest: 9000, host: 9000, auto_correct: true
  config.vm.network "forwarded_port", guest: 8265, host: 8265, auto_correct: true

  # Qdrant
  config.vm.network "forwarded_port", guest: 6333, host: 6333, auto_correct: true
  config.vm.network "forwarded_port", guest: 6334, host: 6334, auto_correct: true

  # ArangoDB
  config.vm.network "forwarded_port", guest: 8529, host: 8529, auto_correct: true

  # Keycloak
  config.vm.network "forwarded_port", guest: 8443, host: 8443, auto_correct: true

  # Postgres
  config.vm.network "forwarded_port", guest: 5432, host: 5432, auto_correct: true

  # RAGApp frontend
  config.vm.network "forwarded_port", guest: 3001, host: 3001, auto_correct: true

  # Model inference ports
  (8501..8505).each do |port|
    config.vm.network "forwarded_port", guest: port, host: port, auto_correct: true
  end

  # === Provisioning ===
  config.vm.provision "shell", privileged: true, inline: <<-SHELL
    set -eux

    # Fix DNS resolution
    echo "nameserver 1.1.1.1" > /etc/resolv.conf
    echo "nameserver 8.8.8.8" >> /etc/resolv.conf

    # Base tools
    apt-get update -y
    apt-get install -y make curl git gh tree ca-certificates gnupg lsb-release software-properties-common unzip build-essential

    # Docker installation
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
