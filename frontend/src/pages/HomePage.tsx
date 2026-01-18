import { Link } from 'react-router-dom'
import { Search, MessageSquare, TrendingUp, Zap } from 'lucide-react'

const features = [
  {
    icon: Search,
    title: '의미 기반 검색',
    description: '자연어로 논문을 검색하세요. AI가 의미를 이해하고 관련 논문을 찾아드립니다.',
    link: '/search',
  },
  {
    icon: MessageSquare,
    title: 'AI 논문 Q&A',
    description: '연구 질문을 하면 관련 논문을 기반으로 신뢰할 수 있는 답변을 제공합니다.',
    link: '/chat',
  },
  {
    icon: TrendingUp,
    title: '연구 트렌드',
    description: '실시간 연구 트렌드를 파악하고 떠오르는 주제를 확인하세요.',
    link: '/trends',
  },
]

export default function HomePage() {
  return (
    <div>
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold liquid-text tracking-tight">
              AI로 더 빠르게,
              <br />
              <span className="bg-gradient-to-r from-amber-300 via-rose-300 to-sky-300 bg-clip-text text-transparent drop-shadow-[0_0_10px_rgba(255,182,193,0.8)]">바이오 연구</span>를 혁신하세요
            </h1>
            <p className="mt-6 text-lg sm:text-xl liquid-text-muted max-w-3xl mx-auto">
              Bio-RAG는 PubMed 논문을 AI로 분석하여 연구자들이 더 빠르게
              인사이트를 얻을 수 있도록 돕습니다.
            </p>
            <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/chat"
                className="glossy-btn-primary inline-flex items-center justify-center px-8 py-4 font-medium"
              >
                <MessageSquare className="mr-2" size={20} />
                AI에게 질문하기
              </Link>
              <Link
                to="/search"
                className="glossy-btn inline-flex items-center justify-center px-8 py-4 font-medium"
              >
                <Search className="mr-2" size={20} />
                논문 검색 시작
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold liquid-text">
              연구를 더 스마트하게
            </h2>
            <p className="mt-4 text-lg liquid-text-muted">
              AI 기술로 논문 분석 시간을 70% 단축하세요
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature) => {
              const Icon = feature.icon
              return (
                <Link
                  key={feature.title}
                  to={feature.link}
                  className="group glossy-panel p-8 hover:scale-105 transition-all duration-300"
                >
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500/50 to-pink-500/50 rounded-2xl flex items-center justify-center mb-6 group-hover:from-purple-500/70 group-hover:to-pink-500/70 transition-all shadow-lg">
                    <Icon className="text-white" size={24} />
                  </div>
                  <h3 className="text-xl font-semibold liquid-text mb-3">
                    {feature.title}
                  </h3>
                  <p className="liquid-text-muted">
                    {feature.description}
                  </p>
                </Link>
              )
            })}
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="glossy-panel p-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-yellow-300 to-orange-300 bg-clip-text text-transparent">35M+</div>
                <div className="mt-2 liquid-text-muted">PubMed 논문</div>
              </div>
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-pink-300 to-purple-300 bg-clip-text text-transparent">&lt;2s</div>
                <div className="mt-2 liquid-text-muted">평균 응답 시간</div>
              </div>
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-purple-300 to-blue-300 bg-clip-text text-transparent">95%</div>
                <div className="mt-2 liquid-text-muted">답변 정확도</div>
              </div>
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text text-transparent">24/7</div>
                <div className="mt-2 liquid-text-muted">AI 지원</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-24 mx-4">
        <div className="max-w-4xl mx-auto glossy-panel p-12 text-center relative overflow-hidden">
          {/* Decorative blobs inside CTA */}
          <div className="absolute -top-10 -right-10 w-40 h-40 blob blob-cyan opacity-50" />
          <div className="absolute -bottom-10 -left-10 w-32 h-32 blob blob-pink opacity-50" />

          <Zap className="mx-auto text-yellow-300 mb-6 relative z-10" size={48} />
          <h2 className="text-3xl font-bold liquid-text mb-4 relative z-10">
            지금 바로 시작하세요
          </h2>
          <p className="text-lg liquid-text-muted mb-8 max-w-2xl mx-auto relative z-10">
            무료로 Bio-RAG를 체험하고 연구 효율성을 높이세요.
            회원가입 없이도 기본 기능을 사용할 수 있습니다.
          </p>
          <Link
            to="/register"
            className="glossy-btn-primary inline-flex items-center justify-center px-8 py-4 font-medium relative z-10"
          >
            무료 회원가입
          </Link>
        </div>
      </div>
    </div>
  )
}
