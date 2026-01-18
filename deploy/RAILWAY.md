# Railway 무료 배포 가이드

Railway는 무료 티어로 월 $5 크레딧을 제공합니다.

## 📋 사전 준비

1. [Railway 계정](https://railway.app/) 생성 (GitHub 연동 권장)
2. OpenAI API Key

## 🚀 배포 방법

### 방법 1: GitHub 연동 (권장)

1. **Railway Dashboard 접속**
   - https://railway.app/dashboard

2. **New Project → Deploy from GitHub repo**

3. **리포지토리 선택**
   - `david1005910/bio-rag-platform` 선택

4. **서비스 추가** (Add Service)

   #### Backend 서비스
   - New Service → GitHub Repo
   - Root Directory: `backend`
   - 환경 변수 설정:
     ```
     OPENAI_API_KEY=sk-xxx
     JWT_SECRET_KEY=your-secret-key
     DATABASE_URL=${{Postgres.DATABASE_URL}}
     REDIS_URL=${{Redis.REDIS_URL}}
     QDRANT_HOST=localhost
     QDRANT_PORT=6333
     APP_ENV=production
     DEBUG=false
     ```

   #### PostgreSQL 추가
   - New Service → Database → PostgreSQL
   - 자동으로 DATABASE_URL 환경변수 생성됨

   #### Redis 추가
   - New Service → Database → Redis
   - 자동으로 REDIS_URL 환경변수 생성됨

   #### Frontend 서비스
   - New Service → GitHub Repo
   - Root Directory: `frontend`
   - 빌드 설정:
     ```
     Build Command: npm run build
     Start Command: npx serve -s dist -l $PORT
     ```
   - 환경 변수:
     ```
     VITE_API_URL=https://your-backend.railway.app/api/v1
     ```

### 방법 2: CLI 배포

```bash
# 1. Railway CLI 로그인
railway login

# 2. 프로젝트 초기화
cd /Users/admin/Documents/david/RAG_Bio2_agent/bio-rag
railway init

# 3. PostgreSQL 추가
railway add --plugin postgresql

# 4. Redis 추가
railway add --plugin redis

# 5. 환경 변수 설정
railway variables set OPENAI_API_KEY=sk-xxx
railway variables set JWT_SECRET_KEY=your-secret
railway variables set APP_ENV=production
railway variables set DEBUG=false

# 6. 배포
railway up
```

## 💰 무료 티어 제한

| 항목 | 제한 |
|------|------|
| 월 크레딧 | $5 |
| 실행 시간 | ~500시간/월 |
| RAM | 512MB (기본) |
| 스토리지 | 1GB |

### 비용 최적화 팁

1. **단일 서비스로 통합**: Backend만 배포하고 Frontend는 Vercel/Netlify 사용
2. **슬립 모드**: 사용하지 않을 때 자동 절전
3. **PostgreSQL 대신 SQLite**: 개발용으로 적합

## 🔧 환경 변수 전체 목록

```env
# Required
OPENAI_API_KEY=sk-your-api-key
JWT_SECRET_KEY=random-32-char-string
DATABASE_URL=${{Postgres.DATABASE_URL}}

# Optional
PUBMED_API_KEY=your-pubmed-key
REDIS_URL=${{Redis.REDIS_URL}}
APP_ENV=production
DEBUG=false
CORS_ORIGINS=["https://your-frontend.railway.app"]
```

## 📊 배포 후 확인

1. **Backend Health Check**
   ```
   https://your-backend.railway.app/health
   ```

2. **API Docs**
   ```
   https://your-backend.railway.app/docs
   ```

3. **Frontend**
   ```
   https://your-frontend.railway.app
   ```

## ⚠️ 트러블슈팅

### 빌드 실패
- Root Directory 설정 확인
- Dockerfile 경로 확인

### 메모리 부족
- Railway Pro 업그레이드 또는
- 단일 서비스로 통합

### 데이터베이스 연결 실패
- DATABASE_URL 환경변수 확인
- PostgreSQL 서비스 상태 확인

## 🔗 유용한 링크

- [Railway Docs](https://docs.railway.app/)
- [Railway Pricing](https://railway.app/pricing)
- [Railway Templates](https://railway.app/templates)
