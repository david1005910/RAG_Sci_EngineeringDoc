import { useState, useEffect } from 'react'
import { Play, Pause, RotateCcw, ChevronRight } from 'lucide-react'

interface PipelineStep {
  id: number
  name: string
  nameKo: string
  description: string
  color: string
  icon: string
  details: string[]
}

const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: 1,
    name: 'Data Collection',
    nameKo: '데이터 수집',
    description: 'PubMed API에서 논문 메타데이터 수집',
    color: '#3B82F6',
    icon: '📥',
    details: [
      'PubMed E-utilities API 호출',
      '논문 제목, 초록, 저자 추출',
      'Rate Limit: 10 req/sec',
    ],
  },
  {
    id: 2,
    name: 'Text Preprocessing',
    nameKo: '텍스트 전처리',
    description: '텍스트 정제 및 청킹',
    color: '#10B981',
    icon: '🔧',
    details: ['특수문자 제거', '참조번호 정규화', '512 토큰 단위 청킹'],
  },
  {
    id: 3,
    name: 'Embedding Generation',
    nameKo: '임베딩 생성',
    description: 'OpenAI API로 벡터 임베딩 생성',
    color: '#8B5CF6',
    icon: '🧮',
    details: ['text-embedding-3-small 모델', '1536 차원 벡터', '배치 처리 (100개씩)'],
  },
  {
    id: 4,
    name: 'Vector Storage',
    nameKo: '벡터 저장',
    description: 'Qdrant에 벡터 인덱싱',
    color: '#F59E0B',
    icon: '💾',
    details: ['Qdrant Vector DB', 'HNSW 인덱스', '메타데이터 저장'],
  },
  {
    id: 5,
    name: 'Query Processing',
    nameKo: '쿼리 처리',
    description: '사용자 질문 처리 및 임베딩',
    color: '#EC4899',
    icon: '❓',
    details: ['한글 → 영어 번역', '쿼리 임베딩 생성', '검색 파라미터 설정'],
  },
  {
    id: 6,
    name: 'Hybrid Search',
    nameKo: '하이브리드 검색',
    description: 'Dense + Sparse 검색 융합',
    color: '#06B6D4',
    icon: '🔍',
    details: ['Dense: 의미 유사도 (70%)', 'Sparse: 키워드 매칭 (30%)', 'Score Fusion'],
  },
  {
    id: 7,
    name: 'Context Building',
    nameKo: '컨텍스트 구성',
    description: '검색 결과로 프롬프트 구성',
    color: '#EF4444',
    icon: '📋',
    details: ['Top-K 문서 선택', '관련성 점수 기반 정렬', '프롬프트 템플릿 적용'],
  },
  {
    id: 8,
    name: 'LLM Generation',
    nameKo: 'LLM 응답 생성',
    description: 'GPT-4로 답변 생성',
    color: '#22C55E',
    icon: '🤖',
    details: ['GPT-4 API 호출', '컨텍스트 기반 응답', '출처 인용 포함'],
  },
]

export default function PipelineAnimation() {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [showDetails, setShowDetails] = useState(true)

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null

    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= PIPELINE_STEPS.length - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 2000)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isPlaying])

  const handleReset = () => {
    setCurrentStep(0)
    setIsPlaying(false)
  }

  const handleStepClick = (index: number) => {
    setCurrentStep(index)
    setIsPlaying(false)
  }

  return (
    <div className="glossy-panel p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <span className="text-2xl">⚡</span>
            RAG 파이프라인
          </h2>
          <p className="text-white/60 text-sm mt-1">
            데이터 수집부터 AI 응답까지의 처리 과정
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg transition-all ${
              isPlaying
                ? 'bg-orange-500/20 text-orange-400 border border-orange-400/30'
                : 'bg-green-500/20 text-green-400 border border-green-400/30'
            }`}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            onClick={handleReset}
            className="p-2 rounded-lg bg-white/10 text-white/70 border border-white/20 hover:bg-white/20 transition-all"
          >
            <RotateCcw size={20} />
          </button>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className={`px-3 py-2 rounded-lg text-sm transition-all ${
              showDetails
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/30'
                : 'bg-white/10 text-white/70 border border-white/20'
            }`}
          >
            상세 {showDetails ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-xs text-white/50 mb-2">
          <span>시작</span>
          <span>
            {currentStep + 1} / {PIPELINE_STEPS.length}
          </span>
          <span>완료</span>
        </div>
        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-500 ease-out rounded-full"
            style={{
              width: `${((currentStep + 1) / PIPELINE_STEPS.length) * 100}%`,
              background: `linear-gradient(90deg, ${PIPELINE_STEPS[0].color}, ${PIPELINE_STEPS[currentStep].color})`,
            }}
          />
        </div>
      </div>

      {/* Pipeline Steps - Desktop: 2 rows, Mobile: Single column */}
      <div className="hidden lg:block">
        {/* Top Row (Steps 1-4) */}
        <div className="flex items-center justify-between mb-4">
          {PIPELINE_STEPS.slice(0, 4).map((step, index) => (
            <div key={step.id} className="flex items-center">
              <StepBox
                step={step}
                isActive={currentStep === index}
                isPast={currentStep > index}
                onClick={() => handleStepClick(index)}
                showDetails={showDetails && currentStep === index}
              />
              {index < 3 && (
                <div className="mx-2">
                  <ChevronRight
                    size={24}
                    className={`transition-all duration-300 ${
                      currentStep > index ? 'text-cyan-400' : 'text-white/20'
                    }`}
                  />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Connector between rows */}
        <div className="flex justify-end pr-[60px] mb-4">
          <div
            className={`w-1 h-8 rounded-full transition-all duration-300 ${
              currentStep >= 4 ? 'bg-cyan-400' : 'bg-white/20'
            }`}
          />
        </div>

        {/* Bottom Row (Steps 5-8) - Reversed order for flow */}
        <div className="flex items-center justify-between flex-row-reverse">
          {PIPELINE_STEPS.slice(4)
            .reverse()
            .map((step, revIndex) => {
              const index = 7 - revIndex
              return (
                <div key={step.id} className="flex items-center">
                  {revIndex < 3 && (
                    <div className="mx-2">
                      <ChevronRight
                        size={24}
                        className={`rotate-180 transition-all duration-300 ${
                          currentStep > index ? 'text-cyan-400' : 'text-white/20'
                        }`}
                      />
                    </div>
                  )}
                  <StepBox
                    step={step}
                    isActive={currentStep === index}
                    isPast={currentStep > index}
                    onClick={() => handleStepClick(index)}
                    showDetails={showDetails && currentStep === index}
                  />
                </div>
              )
            })}
        </div>
      </div>

      {/* Mobile View - Vertical */}
      <div className="lg:hidden space-y-3">
        {PIPELINE_STEPS.map((step, index) => (
          <div key={step.id} className="flex items-start gap-3">
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all duration-300 ${
                  currentStep === index
                    ? 'ring-2 ring-offset-2 ring-offset-slate-900 scale-110 ring-cyan-400'
                    : currentStep > index
                      ? 'opacity-70'
                      : 'opacity-40'
                }`}
                style={{
                  backgroundColor: step.color,
                }}
              >
                {step.icon}
              </div>
              {index < PIPELINE_STEPS.length - 1 && (
                <div
                  className={`w-0.5 h-8 mt-2 transition-all duration-300 ${
                    currentStep > index ? 'bg-cyan-400' : 'bg-white/20'
                  }`}
                />
              )}
            </div>
            <div
              className={`flex-1 pb-4 transition-all duration-300 cursor-pointer ${
                currentStep === index ? 'opacity-100' : 'opacity-50'
              }`}
              onClick={() => handleStepClick(index)}
            >
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white">{step.nameKo}</span>
                <span className="text-xs text-white/50">Step {step.id}</span>
              </div>
              <p className="text-sm text-white/70 mt-1">{step.description}</p>
              {showDetails && currentStep === index && (
                <div className="mt-2 space-y-1">
                  {step.details.map((detail, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs text-white/60">
                      <div
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: step.color }}
                      />
                      {detail}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Current Step Details Panel (Desktop) */}
      {showDetails && (
        <div className="hidden lg:block mt-6 p-4 rounded-xl border border-white/10 bg-white/5">
          <div className="flex items-start gap-4">
            <div
              className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl shrink-0"
              style={{ backgroundColor: PIPELINE_STEPS[currentStep].color }}
            >
              {PIPELINE_STEPS[currentStep].icon}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="font-semibold text-white text-lg">
                  {PIPELINE_STEPS[currentStep].nameKo}
                </h3>
                <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/60">
                  Step {PIPELINE_STEPS[currentStep].id}
                </span>
              </div>
              <p className="text-white/80 text-sm mb-3">{PIPELINE_STEPS[currentStep].description}</p>
              <div className="flex flex-wrap gap-2">
                {PIPELINE_STEPS[currentStep].details.map((detail, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 rounded-full text-xs font-medium"
                    style={{
                      backgroundColor: `${PIPELINE_STEPS[currentStep].color}20`,
                      color: PIPELINE_STEPS[currentStep].color,
                      border: `1px solid ${PIPELINE_STEPS[currentStep].color}40`,
                    }}
                  >
                    {detail}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Code Example */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-white/60">코드 예시</span>
          <span className="text-xs text-white/40">{PIPELINE_STEPS[currentStep].name}</span>
        </div>
        <CodeExample step={PIPELINE_STEPS[currentStep]} />
      </div>
    </div>
  )
}

function StepBox({
  step,
  isActive,
  isPast,
  onClick,
}: {
  step: PipelineStep
  isActive: boolean
  isPast: boolean
  onClick: () => void
  showDetails?: boolean
}) {
  return (
    <div
      onClick={onClick}
      className={`relative cursor-pointer transition-all duration-300 ${
        isActive ? 'scale-105 z-10' : isPast ? 'opacity-70' : 'opacity-40'
      }`}
    >
      <div
        className={`w-32 h-28 rounded-xl p-3 flex flex-col items-center justify-center transition-all duration-300 ${
          isActive ? 'ring-2 ring-offset-2 ring-offset-slate-900 ring-cyan-400' : ''
        }`}
        style={{
          backgroundColor: isActive ? step.color : `${step.color}30`,
          borderColor: step.color,
          border: `2px solid ${step.color}`,
        }}
      >
        <span className="text-2xl mb-1">{step.icon}</span>
        <span className="text-white text-xs font-semibold text-center">{step.nameKo}</span>
        <span className="text-white/60 text-[10px]">Step {step.id}</span>
      </div>

      {/* Pulse animation for active step */}
      {isActive && (
        <div
          className="absolute inset-0 rounded-xl animate-ping opacity-30"
          style={{ backgroundColor: step.color }}
        />
      )}
    </div>
  )
}

function CodeExample({ step }: { step: PipelineStep }) {
  const codeExamples: Record<number, string> = {
    1: `# PubMed API 데이터 수집
async def fetch_papers(query: str, max_results: int = 100):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results}
    response = await httpx.get(url, params=params)
    return parse_pubmed_response(response.json())`,

    2: `# 텍스트 전처리 및 청킹
def preprocess_text(text: str) -> list[str]:
    # 특수문자 제거 및 정규화
    cleaned = re.sub(r'\\[\\d+\\]', '', text)  # 참조번호 제거
    cleaned = re.sub(r'[^\\w\\s.-]', '', cleaned)

    # 512 토큰 단위로 청킹
    chunks = split_into_chunks(cleaned, max_tokens=512)
    return chunks`,

    3: `# OpenAI 임베딩 생성
async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=1536
    )
    return [item.embedding for item in response.data]`,

    4: `# Qdrant 벡터 저장
async def store_vectors(embeddings: list, metadata: list):
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=emb,
            payload=meta
        )
        for emb, meta in zip(embeddings, metadata)
    ]
    await qdrant.upsert(collection_name="papers", points=points)`,

    5: `# 쿼리 처리
async def process_query(question: str) -> dict:
    # 한글 → 영어 번역 (필요시)
    if detect_language(question) == "ko":
        question = await translate(question, "ko", "en")

    # 쿼리 임베딩 생성
    query_embedding = await generate_embedding(question)
    return {"embedding": query_embedding, "original": question}`,

    6: `# 하이브리드 검색
async def hybrid_search(query_emb, query_text, top_k=10):
    # Dense search (70%)
    dense_results = await qdrant.search(
        collection="papers", query_vector=query_emb, limit=top_k
    )
    # Sparse search (30%)
    sparse_results = await bm25_search(query_text, limit=top_k)

    # Score fusion
    return fuse_scores(dense_results, sparse_results, weights=[0.7, 0.3])`,

    7: `# 컨텍스트 구성
def build_context(search_results: list, max_tokens: int = 4000) -> str:
    context_parts = []
    for result in search_results[:5]:  # Top-K 선택
        context_parts.append(f'''
        [PMID: {result.pmid}] {result.title}
        {result.abstract[:500]}...
        ''')
    return "\\n".join(context_parts)`,

    8: `# LLM 응답 생성
async def generate_answer(question: str, context: str) -> str:
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content`,
  }

  return (
    <div className="rounded-lg overflow-hidden bg-slate-900/80 border border-white/10">
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 border-b border-white/10">
        <div className="w-3 h-3 rounded-full bg-red-500" />
        <div className="w-3 h-3 rounded-full bg-yellow-500" />
        <div className="w-3 h-3 rounded-full bg-green-500" />
        <span className="ml-2 text-xs text-white/50">{step.name.toLowerCase().replace(/ /g, '_')}.py</span>
      </div>
      <pre className="p-4 overflow-x-auto text-xs leading-relaxed">
        <code className="text-cyan-300/90">{codeExamples[step.id]}</code>
      </pre>
    </div>
  )
}
