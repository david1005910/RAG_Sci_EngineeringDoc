# Bio-RAG AWS EC2 Deployment Guide

이 가이드는 Bio-RAG 플랫폼을 AWS EC2에 배포하는 방법을 설명합니다.

## 📋 사전 요구사항

- AWS 계정
- 도메인 (선택사항, SSL용)
- OpenAI API Key
- PubMed API Key (선택사항)

## 🚀 빠른 시작

### Step 1: EC2 인스턴스 생성

1. **AWS Console** → EC2 → Launch Instance

2. **인스턴스 설정:**
   - **Name**: bio-rag-server
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Instance type**: t3.medium (최소), t3.large (권장)
   - **Storage**: 30GB gp3 (최소)

3. **Security Group 설정:**
   | Port | Protocol | Source | Description |
   |------|----------|--------|-------------|
   | 22 | TCP | Your IP | SSH |
   | 80 | TCP | 0.0.0.0/0 | HTTP |
   | 443 | TCP | 0.0.0.0/0 | HTTPS |

4. **Key Pair**: 새로 생성하거나 기존 키 사용

### Step 2: 서버 접속 및 초기 설정

```bash
# SSH 접속
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>

# 설정 스크립트 다운로드 및 실행
curl -sSL https://raw.githubusercontent.com/david1005910/bio-rag-platform/main/deploy/ec2-setup.sh | bash

# 재접속 (docker 그룹 적용)
exit
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3: 환경 변수 설정

```bash
cd /opt/bio-rag
nano .env
```

필수 환경 변수:
```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key
POSTGRES_PASSWORD=your-secure-database-password
JWT_SECRET_KEY=your-random-jwt-secret-key

# Optional
PUBMED_API_KEY=your-pubmed-api-key
DOMAIN=your-domain.com
```

**JWT_SECRET_KEY 생성:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### Step 4: 배포 실행

```bash
# 배포 스크립트 실행
chmod +x deploy/deploy.sh
./deploy/deploy.sh

# 또는 수동 실행
docker compose -f deploy/docker-compose.prod.yml up -d --build
```

### Step 5: SSL 설정 (선택사항)

도메인이 있는 경우:

```bash
# 도메인 DNS A 레코드를 EC2 Public IP로 설정 후
chmod +x deploy/ssl-setup.sh
./deploy/ssl-setup.sh your-domain.com your-email@domain.com
```

## 📊 서비스 확인

```bash
# 컨테이너 상태 확인
docker compose -f deploy/docker-compose.prod.yml ps

# 로그 확인
docker compose -f deploy/docker-compose.prod.yml logs -f

# 특정 서비스 로그
docker compose -f deploy/docker-compose.prod.yml logs -f backend

# Health check
curl http://localhost/health
```

## 🔧 유지보수

### 업데이트 배포

```bash
cd /opt/bio-rag
git pull origin main
./deploy/deploy.sh
```

### 백업

```bash
# PostgreSQL 백업
docker compose -f deploy/docker-compose.prod.yml exec postgres pg_dump -U bio_rag bio_rag > backup.sql

# 볼륨 백업
docker run --rm -v bio-rag_postgres_data:/data -v $(pwd):/backup alpine tar cvf /backup/postgres_backup.tar /data
```

### 복구

```bash
# PostgreSQL 복구
docker compose -f deploy/docker-compose.prod.yml exec -T postgres psql -U bio_rag bio_rag < backup.sql
```

## 📈 모니터링

### 리소스 사용량
```bash
# Docker 통계
docker stats

# 디스크 사용량
df -h

# 메모리 사용량
free -m
```

### 로그 위치
- Backend: `docker logs bio-rag-backend`
- Nginx: `docker logs bio-rag-nginx`
- PostgreSQL: `docker logs bio-rag-postgres`

## ⚠️ 트러블슈팅

### 1. 컨테이너가 시작되지 않음
```bash
# 로그 확인
docker compose -f deploy/docker-compose.prod.yml logs

# 컨테이너 재시작
docker compose -f deploy/docker-compose.prod.yml restart
```

### 2. 메모리 부족
```bash
# 스왑 메모리 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. 포트 충돌
```bash
# 사용 중인 포트 확인
sudo lsof -i :80
sudo lsof -i :443
```

## 💰 비용 최적화

### 권장 인스턴스
| 용도 | 인스턴스 | vCPU | RAM | 월 비용 (예상) |
|------|----------|------|-----|----------------|
| 개발/테스트 | t3.medium | 2 | 4GB | ~$30 |
| 소규모 | t3.large | 2 | 8GB | ~$60 |
| 프로덕션 | t3.xlarge | 4 | 16GB | ~$120 |

### 비용 절감 팁
- Spot Instance 사용 (최대 90% 할인)
- Reserved Instance 1년 약정 (최대 40% 할인)
- 개발 환경은 사용하지 않을 때 중지

## 📞 지원

문제가 발생하면 GitHub Issues에 등록해주세요:
https://github.com/david1005910/bio-rag-platform/issues
