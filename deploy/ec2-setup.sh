#!/bin/bash
# Bio-RAG EC2 Setup Script
# Run this script on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "=========================================="
echo "Bio-RAG EC2 Setup Script"
echo "=========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "[2/6] Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER

# Install additional tools
echo "[3/6] Installing additional tools..."
sudo apt-get install -y git htop vim

# Create app directory
echo "[4/6] Creating application directory..."
sudo mkdir -p /opt/bio-rag
sudo chown $USER:$USER /opt/bio-rag

# Clone repository
echo "[5/6] Cloning repository..."
cd /opt/bio-rag
if [ -d ".git" ]; then
    git pull origin main
else
    git clone https://github.com/david1005910/bio-rag-platform.git .
fi

# Create environment file
echo "[6/6] Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "=========================================="
    echo "IMPORTANT: Edit /opt/bio-rag/.env file"
    echo "Set your API keys:"
    echo "  - OPENAI_API_KEY"
    echo "  - PUBMED_API_KEY (optional)"
    echo "  - POSTGRES_PASSWORD"
    echo "  - JWT_SECRET_KEY"
    echo "=========================================="
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Log out and log back in (for docker group)"
echo "2. Edit /opt/bio-rag/.env with your API keys"
echo "3. Run: cd /opt/bio-rag && docker compose up -d"
echo "=========================================="
