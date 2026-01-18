#!/bin/bash
# SSL Certificate Setup with Let's Encrypt

set -e

DOMAIN=${1:-""}

if [ -z "$DOMAIN" ]; then
    echo "Usage: ./ssl-setup.sh <your-domain.com>"
    echo "Example: ./ssl-setup.sh bio-rag.example.com"
    exit 1
fi

EMAIL=${2:-"admin@$DOMAIN"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "SSL Certificate Setup"
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo "=========================================="

# Create SSL directory
mkdir -p "$SCRIPT_DIR/ssl"

# Create temporary self-signed certificate (for initial nginx start)
echo "[1/4] Creating temporary self-signed certificate..."
openssl req -x509 -nodes -days 1 -newkey rsa:2048 \
    -keyout "$SCRIPT_DIR/ssl/privkey.pem" \
    -out "$SCRIPT_DIR/ssl/fullchain.pem" \
    -subj "/CN=$DOMAIN"

# Start nginx with temporary certificate
echo "[2/4] Starting nginx..."
docker compose -f "$SCRIPT_DIR/docker-compose.prod.yml" up -d nginx

# Wait for nginx to start
sleep 5

# Request Let's Encrypt certificate
echo "[3/4] Requesting Let's Encrypt certificate..."
docker compose -f "$SCRIPT_DIR/docker-compose.prod.yml" run --rm certbot \
    certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN"

# Copy certificates
echo "[4/4] Installing certificates..."
cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem "$SCRIPT_DIR/ssl/"
cp /etc/letsencrypt/live/$DOMAIN/privkey.pem "$SCRIPT_DIR/ssl/"

# Reload nginx
docker compose -f "$SCRIPT_DIR/docker-compose.prod.yml" exec nginx nginx -s reload

echo ""
echo "=========================================="
echo "SSL setup complete!"
echo "Your site is now available at: https://$DOMAIN"
echo ""
echo "Certificates will auto-renew via certbot container"
echo "=========================================="
