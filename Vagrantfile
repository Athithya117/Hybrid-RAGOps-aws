Vagrant.configure("2") do |config|
  config.vm.boot_timeout = 3600
  config.vm.box = "ubuntu/jammy64"
  config.vm.box_version = "20241002.0.0"

  # Use rsync with auto sync enabled if you want:
  config.vm.synced_folder ".", "/vagrant", type: "rsync", rsync__auto: true

  config.ssh.insert_key = false

  config.vm.network "private_network", ip: "192.168.56.18"

  # === Port forwards ===
  config.vm.network "forwarded_port", guest: 80, host: 80, auto_correct: true
  config.vm.network "forwarded_port", guest: 443, host: 443, auto_correct: true
  config.vm.network "forwarded_port", guest: 5000, host: 5000, auto_correct: true
  config.vm.network "forwarded_port", guest: 9090, host: 9090, auto_correct: true
  config.vm.network "forwarded_port", guest: 3000, host: 3000, auto_correct: true
  config.vm.network "forwarded_port", guest: 9093, host: 9093, auto_correct: true
  config.vm.network "forwarded_port", guest: 8000, host: 8000, auto_correct: true
  config.vm.network "forwarded_port", guest: 9000, host: 9000, auto_correct: true
  config.vm.network "forwarded_port", guest: 8265, host: 8265, auto_correct: true
  config.vm.network "forwarded_port", guest: 6333, host: 6333, auto_correct: true
  config.vm.network "forwarded_port", guest: 6334, host: 6334, auto_correct: true
  config.vm.network "forwarded_port", guest: 8529, host: 8529, auto_correct: true
  config.vm.network "forwarded_port", guest: 8443, host: 8443, auto_correct: true
  config.vm.network "forwarded_port", guest: 5432, host: 5432, auto_correct: true
  config.vm.network "forwarded_port", guest: 3001, host: 3001, auto_correct: true
  config.vm.network "forwarded_port", guest: 8501, host: 8501, auto_correct: true
  config.vm.network "forwarded_port", guest: 8502, host: 8502, auto_correct: true
  config.vm.network "forwarded_port", guest: 8503, host: 8503, auto_correct: true
  config.vm.network "forwarded_port", guest: 8504, host: 8504, auto_correct: true
  config.vm.network "forwarded_port", guest: 8505, host: 8505, auto_correct: true

  config.vm.provider "virtualbox" do |vb|
    vb.memory = 11000
    vb.cpus = 6
    vb.gui = false
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
  end

  config.vm.provision "shell", inline: <<-SHELL

    # === Disable buggy DNS resolver & set static nameservers ===
    sudo systemctl disable --now systemd-resolved || true
    sudo apt-get install -y resolvconf
    echo -e "nameserver 1.1.1.1\\nnameserver 8.8.8.8" | sudo tee /etc/resolvconf/resolv.conf.d/base
    sudo resolvconf -u

    # === Install base tools ===
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \\
      build-essential curl git make gh ca-certificates gnupg lsb-release unzip

    # === Install Docker (pinned version) ===
    DOCKER_VERSION="5:27.5.1-1~ubuntu.22.04~jammy"
    sudo mkdir -p /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      sudo chmod a+r /etc/apt/keyrings/docker.gpg
    fi
    echo \\
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \\
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \\
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    if ! dpkg -l | grep -q "docker-ce.*${DOCKER_VERSION}"; then
      sudo apt-get install -y \\
        docker-ce=${DOCKER_VERSION} \\
        docker-ce-cli=${DOCKER_VERSION} \\
        containerd.io
    fi
    sudo apt-mark hold docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker vagrant
    sudo systemctl enable docker

  SHELL
end
