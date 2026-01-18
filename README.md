# Bio-RAG

<div align="center">

![Bio-RAG Logo](https://img.shields.io/badge/Bio--RAG-AI%20Research%20Platform-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTkuNSAyQTIuNSAyLjUgMCAwIDEgMTIgNC41djE1YTIuNSAyLjUgMCAwIDEtNC45NS40NGwtLjUtMi43MkEyLjUgMi41IDAgMCAxIDkuNSAyWiIvPjxwYXRoIGQ9Ik0xNC41IDJBMS41IDEuNSAwIDAgMSAxNiAzLjV2MTdhMS41IDEuNSAwIDAgMS0zIDBWMy41QTEuNSAxLjUgMCAwIDEgMTQuNSAyWiIvPjwvc3ZnPg==)

**Biomedical Research AI-Guided Analytics**

바이오 연구자를 위한 AI 기반 논문 분석 및 인사이트 도출 플랫폼

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)

[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-7+-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)

[Demo](#demo) • [Features](#features) • [Quick Start](#quick-start) • [Documentation](#api-documentation)

</div>

---

## Demo

### Homepage
<div align="center">
<img src="docs/screenshots/home.png" alt="Bio-RAG Homepage" width="800">
<p><em>AI 기반 바이오메디컬 연구 플랫폼 메인 화면</em></p>
</div>

### Semantic Search
<div align="center">
<img src="docs/screenshots/search.png" alt="Paper Search" width="800">
<p><em>자연어 기반 의미 검색으로 관련 논문을 빠르게 찾기</em></p>
</div>

### AI Chat Q&A
<div align="center">
<img src="docs/screenshots/chat.png" alt="AI Chat" width="800">
<p><em>RAG 기반 AI 질의응답 - 논문 근거와 함께 답변 제공</em></p>
</div>

### Research Trends
<div align="center">
<img src="docs/screenshots/trends.png" alt="Research Trends" width="800">
<p><em>연구 트렌드 대시보드 - 핫토픽, 키워드 추이, 분야별 분포</em></p>
</div>

### Paper Detail
<div align="center">
<img src="docs/screenshots/paper-detail.png" alt="Paper Detail" width="800">
<p><em>논문 상세 정보 및 유사 논문 추천</em></p>
</div>

---

## Features

| Feature | Description |
|---------|-------------|
| **AI 논문 Q&A** | RAG (Retrieval-Augmented Generation) 기반 자연어 질의응답. VectorDB에서 관련 논문을 검색하고 GPT-4가 근거 기반 답변 생성 |
| **하이브리드 검색** | OpenAI Embedding (Dense) + SPLADE (Sparse) 하이브리드 검색으로 정확도와 재현율 동시 향상 |
| **의미 기반 검색** | Qdrant 벡터 DB + 1536차원 OpenAI 임베딩을 활용한 시맨틱 검색. 키워드가 아닌 의미로 논문 탐색 |
| **한글 검색 지원** | 한글 쿼리 자동 번역 및 PubMed 검색, 결과 한글 요약 기능 |
| **유사 논문 추천** | 코사인 유사도 기반 관련 논문 자동 추천. 연구 범위를 효율적으로 확장 |
| **연구 트렌드 분석** | 시계열 키워드 분석, 핫 토픽, 신흥 연구 분야 시각화 대시보드 |
| **개인 라이브러리** | 관심 논문 저장, 태그 관리, 메모 기능. JSON 파일 기반 영구 저장 |
| **사용자 인증** | JWT 기반 회원가입/로그인, 영구 사용자 저장 |

---

## Tech Stack

<table>
<tr>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg" width="48" height="48" alt="React" />
<br>React 18
</td>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/typescript/typescript-original.svg" width="48" height="48" alt="TypeScript" />
<br>TypeScript
</td>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tailwindcss/tailwindcss-original.svg" width="48" height="48" alt="TailwindCSS" />
<br>TailwindCSS
</td>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48" height="48" alt="Python" />
<br>Python 3.11
</td>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" width="48" height="48" alt="FastAPI" />
<br>FastAPI
</td>
</tr>
<tr>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg" width="48" height="48" alt="PostgreSQL" />
<br>PostgreSQL
</td>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg" width="48" height="48" alt="Redis" />
<br>Redis
</td>
<td align="center" width="120">
<img src="https://qdrant.tech/images/logo_with_text.svg" width="48" height="48" alt="Qdrant" />
<br>Qdrant
</td>
<td align="center" width="120">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="48" height="48" alt="Docker" />
<br>Docker
</td>
<td align="center" width="120">
<img src="https://www.langchain.com/favicon.ico" width="48" height="48" alt="LangChain" />
<br>LangChain
</td>
</tr>
</table>

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18 + TypeScript + TailwindCSS + React Query + Zustand + Recharts |
| **Backend** | Python 3.11 + FastAPI + SQLAlchemy + Pydantic + Celery |
| **AI/ML** | OpenAI GPT-4 + OpenAI Embeddings (1536d) + SPLADE Sparse Search + LangChain |
| **Vector Search** | Qdrant (Dense) + SPLADE BM25 (Sparse) Hybrid Search (70:30 가중치) |
| **Database** | PostgreSQL (metadata) + Qdrant (vectors) + Redis (cache/queue) |
| **Infrastructure** | Docker Compose + Nginx + AWS (optional) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (optional)
- OpenAI API Key
- PubMed API Key (optional)

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/david1005910/bio-rag-platform.git
cd bio-rag-platform

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY (required)
# - PUBMED_API_KEY (optional)
# - POSTGRES_PASSWORD
# - JWT_SECRET_KEY

# Start all services
docker compose up -d --build

# Check status
docker compose ps

# View logs
docker compose logs -f

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Qdrant Dashboard: http://localhost:6333/dashboard
```

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| frontend | 3000 | React + Nginx |
| backend | 8000 | FastAPI |
| postgres | 5432 | PostgreSQL 16 |
| redis | 6379 | Redis 7 |
| qdrant | 6333/6334 | Vector Database |
| celery-worker | - | Background Tasks |

### Option 2: Local Development

```bash
# Clone repository
git clone https://github.com/david1005910/bio-rag-platform.git
cd bio-rag-platform/bio-rag

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys

# Start backend
uvicorn src.main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000/api/v1" > .env
npm run dev
```

### Environment Variables

Create `backend/.env` from `.env.example`:

```env
# Required
OPENAI_API_KEY=your-openai-api-key
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/bio_rag

# Optional
PUBMED_API_KEY=your-pubmed-api-key
QDRANT_HOST=localhost
REDIS_URL=redis://localhost:6379/0
```

---

## Project Structure

```
bio-rag/
├── docker-compose.yml       # Docker orchestration
├── .env.example             # Environment template
├── backend/                 # FastAPI Backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── src/
│   │   ├── api/v1/         # REST API endpoints
│   │   │   ├── auth.py     # JWT Authentication
│   │   │   ├── search.py   # PubMed search + translation
│   │   │   ├── chat.py     # RAG-based AI Q&A
│   │   │   ├── library.py  # User paper library
│   │   │   ├── trends.py   # Research trends
│   │   │   └── vectordb.py # Hybrid vector search
│   │   ├── core/           # Config, DB, Security
│   │   ├── data/           # Persistent storage
│   │   │   ├── users.py    # User store (JSON)
│   │   │   └── library_store.py  # Library store (JSON)
│   │   ├── models/         # SQLAlchemy models
│   │   ├── services/       # Business logic
│   │   │   ├── pubmed.py   # PubMed API client
│   │   │   ├── embedding/  # Vector embeddings
│   │   │   └── search/     # Semantic search
│   │   └── tasks/          # Celery async tasks
│   ├── data/               # Runtime data (gitignored)
│   │   ├── users.json      # User data
│   │   ├── library.json    # Library data
│   │   └── qdrant/         # Vector database
│   └── tests/              # Pytest tests
├── frontend/               # React Frontend
│   ├── Dockerfile
│   ├── nginx.conf          # Production nginx config
│   └── src/
│       ├── pages/          # Route pages
│       ├── components/     # UI components
│       ├── services/       # API client
│       └── store/          # Zustand state
└── docs/                   # Documentation
    └── screenshots/        # App screenshots
```

---

## Architecture

### Hybrid Search System

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
│                  "CRISPR gene therapy"                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌──────────┴──────────┐
          ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│   Dense Search  │   │  Sparse Search  │
│  (OpenAI 1536d) │   │  (SPLADE BM25)  │
│                 │   │                 │
│  Semantic       │   │  Keyword        │
│  Similarity     │   │  Matching       │
└────────┬────────┘   └────────┬────────┘
         │                     │
         │    70%              │    30%
         └──────────┬──────────┘
                    ▼
         ┌─────────────────┐
         │  Score Fusion   │
         │  (Weighted Sum) │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  Top-K Results  │
         └─────────────────┘
```

### Data Flow

```
PubMed Search → Auto-save to VectorDB → Hybrid Search → AI Chat (RAG)
                      │
                      ▼
              ┌───────────────┐
              │    Qdrant     │
              │  (Local/Docker)│
              │  1536d vectors │
              └───────────────┘
```

---

## API Documentation

After starting the backend, access interactive API docs:

| Documentation | URL |
|--------------|-----|
| **Swagger UI** | http://localhost:8000/docs |
| **ReDoc** | http://localhost:8000/redoc |
| **OpenAPI JSON** | http://localhost:8000/openapi.json |

### Key Endpoints

```
# Search
POST /api/v1/search              # PubMed paper search (한글 지원)
POST /api/v1/translate           # Korean → English translation
POST /api/v1/summarize           # Paper summarization

# AI Chat
POST /api/v1/chat/query          # RAG-based Q&A with VectorDB

# VectorDB
POST /api/v1/vectordb/papers/save    # Save papers to VectorDB
POST /api/v1/vectordb/search         # Hybrid search (Dense + Sparse)
GET  /api/v1/vectordb/stats          # Collection statistics

# Library
POST /api/v1/library/papers      # Save paper to library
GET  /api/v1/library/papers      # Get saved papers

# Trends
GET  /api/v1/trends/hot          # Hot topics
GET  /api/v1/trends/keywords     # Keyword trends
POST /api/v1/trends/analyze      # Trend analysis

# Auth
POST /api/v1/auth/register       # User registration
POST /api/v1/auth/login          # User login
```

---

## Development

### Running Tests

```bash
cd backend
pytest -v
```

### Code Quality

```bash
# Backend
pip install ruff black
ruff check src/
black src/

# Frontend
npm run lint
npm run type-check
```

---

## Roadmap

### Completed
- [x] Core RAG Q&A functionality
- [x] Semantic paper search
- [x] Research trends dashboard
- [x] Similar paper recommendations
- [x] Real PubMed API integration
- [x] User authentication (JWT)
- [x] Hybrid search (Dense + Sparse)
- [x] VectorDB (Qdrant) integration
- [x] Korean language support (translation & summarization)
- [x] Persistent user & library storage
- [x] Docker Compose deployment

### Planned
- [ ] Paper PDF upload & parsing
- [ ] PMC full-text search
- [ ] Mobile responsive design
- [ ] Export to citation formats (BibTeX, RIS)
- [ ] Batch paper processing
- [ ] Custom embedding fine-tuning

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with AI for Biomedical Researchers**

Made with ❤️ by [david1005910](https://github.com/david1005910)

</div>
