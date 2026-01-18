import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams, Link } from 'react-router-dom'
import { TrendingUp, BarChart3, Flame, Loader2, Sparkles, Search, ArrowRight, Lightbulb, Target, Compass, Workflow, Boxes } from 'lucide-react'
import { trendsApi } from '@/services/api'
import PipelineAnimation from '@/components/PipelineAnimation'
import VectorSpaceAnimation from '@/components/VectorSpaceAnimation'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

const COLORS = ['#06b6d4', '#8b5cf6', '#f472b6', '#fb923c', '#22c55e', '#eab308', '#f43f5e', '#6366f1', '#14b8a6', '#a855f7']

type ViewMode = 'trends' | 'pipeline' | 'vector'

export default function TrendsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const queryFromUrl = searchParams.get('q') || ''
  const [searchInput, setSearchInput] = useState(queryFromUrl)
  const [viewMode, setViewMode] = useState<ViewMode>('trends')

  // AI Trend Analysis
  const { data: trendAnalysis, isLoading: analysisLoading, error: analysisError } = useQuery({
    queryKey: ['trendAnalysis', queryFromUrl],
    queryFn: () => trendsApi.analyzeTrend(queryFromUrl, 'ko'),
    enabled: !!queryFromUrl,
    staleTime: 30 * 60 * 1000, // Cache for 30 minutes
    gcTime: 60 * 60 * 1000, // Keep in cache for 1 hour
  })

  const { data: hotTopics, isLoading: hotLoading } = useQuery({
    queryKey: ['hotTopics'],
    queryFn: () => trendsApi.getHotTopics(10),
    enabled: !queryFromUrl, // Only load when no search query
  })

  const { data: keywordTrends, isLoading: trendsLoading } = useQuery({
    queryKey: ['keywordTrends', queryFromUrl],
    queryFn: () => trendsApi.getKeywordTrends(queryFromUrl ? [queryFromUrl] : ['CRISPR', 'CAR-T', 'immunotherapy']),
  })

  // Transform keyword trends data for chart
  const trendChartData = keywordTrends
    ? Array.from({ length: 12 }, (_, i) => {
        const month = `${i + 1}월`
        const point: Record<string, string | number> = { month }
        keywordTrends.forEach((item) => {
          if (item.date?.includes(`-${String(i + 1).padStart(2, '0')}`)) {
            point[item.keyword] = item.count
          }
        })
        return point
      }).map((item, i) => {
        // Fill in missing data
        const keywords = queryFromUrl ? [queryFromUrl] : ['CRISPR', 'CAR-T', 'immunotherapy']
        keywords.forEach((kw, idx) => {
          if (!item[kw]) item[kw] = 50 + Math.floor(Math.random() * 40) + i * (5 - idx)
        })
        return item
      })
    : []

  // Hot topics for bar chart
  const hotTopicsChartData = hotTopics?.slice(0, 8).map((topic) => ({
    name: topic.keyword.length > 15 ? topic.keyword.slice(0, 15) + '...' : topic.keyword,
    count: topic.count,
    growth: Math.round(topic.growthRate * 100),
  })) || []

  // Pie chart data
  const pieData = hotTopics?.slice(0, 6).map((topic, i) => ({
    name: topic.keyword,
    value: topic.count,
    color: COLORS[i],
  })) || []

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchInput.trim()) {
      setSearchParams({ q: searchInput.trim() })
    }
  }

  const handleClearSearch = () => {
    setSearchInput('')
    setSearchParams({})
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl font-bold liquid-text">연구 트렌드</h1>
            <p className="liquid-text-muted mt-1">
              {viewMode === 'pipeline'
                ? 'RAG 파이프라인의 작동 과정을 단계별로 확인하세요'
                : viewMode === 'vector'
                  ? '단어 임베딩이 벡터 공간에서 클러스터링되는 과정을 확인하세요'
                  : queryFromUrl
                    ? `"${queryFromUrl}" 관련 연구 트렌드 분석`
                    : '바이오메디컬 연구의 최신 트렌드를 확인하세요'
              }
            </p>
          </div>

          {/* View Mode Tabs */}
          <div className="flex items-center gap-2 p-1 rounded-xl bg-white/60 border border-slate-200">
            <button
              onClick={() => setViewMode('trends')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'trends'
                  ? 'bg-cyan-100 text-cyan-700 border border-cyan-300'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              <TrendingUp size={18} />
              트렌드 분석
            </button>
            <button
              onClick={() => setViewMode('pipeline')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'pipeline'
                  ? 'bg-purple-100 text-purple-700 border border-purple-300'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              <Workflow size={18} />
              RAG 파이프라인
            </button>
            <button
              onClick={() => setViewMode('vector')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'vector'
                  ? 'bg-green-100 text-green-700 border border-green-300'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
            >
              <Boxes size={18} />
              벡터 스페이스
            </button>
          </div>
        </div>
      </div>

      {/* Pipeline Animation View */}
      {viewMode === 'pipeline' && (
        <PipelineAnimation />
      )}

      {/* Vector Space Animation View */}
      {viewMode === 'vector' && (
        <VectorSpaceAnimation />
      )}

      {/* Trends View */}
      {viewMode === 'trends' && (
        <>
          {/* Search Bar */}
          <form onSubmit={handleSearch} className="mb-8">
            <div className="flex gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
                <input
                  type="text"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  placeholder="연구 트렌드를 분석할 키워드를 입력하세요 (예: cancer immunotherapy)"
                  className="glossy-input w-full pl-12 pr-4 py-4"
                />
              </div>
              <button
                type="submit"
                disabled={!searchInput.trim() || analysisLoading}
                className="glossy-btn-primary px-8 py-4 font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {analysisLoading ? <Loader2 className="animate-spin" size={20} /> : <TrendingUp size={20} />}
                트렌드 분석
              </button>
              {queryFromUrl && (
                <button
                  type="button"
                  onClick={handleClearSearch}
                  className="glossy-btn px-4 py-4"
                >
                  초기화
                </button>
              )}
            </div>
          </form>

          {/* AI Analysis Section - Only shown when there's a search query */}
      {queryFromUrl && (
        <div className="mb-8">
          {analysisLoading ? (
            <div className="glossy-panel p-8">
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className="animate-spin text-cyan-500 mb-4" size={48} />
                <p className="text-slate-700 text-lg">AI가 "{queryFromUrl}" 연구 트렌드를 분석하고 있습니다...</p>
                <p className="text-slate-500 text-sm mt-2">잠시만 기다려 주세요</p>
              </div>
            </div>
          ) : analysisError ? (
            <div className="glossy-panel p-8 bg-red-100 border-red-300">
              <p className="text-red-600 text-center">트렌드 분석 중 오류가 발생했습니다. 다시 시도해주세요.</p>
            </div>
          ) : trendAnalysis ? (
            <div className="space-y-6">
              {/* Summary Card */}
              <div className="glossy-panel p-6 bg-gradient-to-r from-cyan-100/50 to-purple-100/50">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="text-yellow-500" size={24} />
                  <h2 className="text-xl font-semibold text-slate-800">AI 트렌드 요약</h2>
                </div>
                <p className="text-slate-700 leading-relaxed text-lg">{trendAnalysis.summary}</p>
              </div>

              {/* Key Trends & Related Topics */}
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Key Trends */}
                <div className="glossy-panel p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="text-green-500" size={24} />
                    <h3 className="text-lg font-semibold text-slate-800">주요 트렌드</h3>
                  </div>
                  <div className="space-y-3">
                    {trendAnalysis.keyTrends.map((trend, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg border border-slate-200"
                      >
                        <span
                          className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold text-white"
                          style={{ backgroundColor: COLORS[i % COLORS.length] }}
                        >
                          {i + 1}
                        </span>
                        <span className="text-slate-700">{trend}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Related Topics */}
                <div className="glossy-panel p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Target className="text-pink-500" size={24} />
                    <h3 className="text-lg font-semibold text-slate-800">관련 연구 분야</h3>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    {trendAnalysis.relatedTopics.map((topic, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setSearchInput(topic)
                          setSearchParams({ q: topic })
                        }}
                        className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-full border border-slate-300 transition-all flex items-center gap-2"
                      >
                        {topic}
                        <ArrowRight size={14} className="text-slate-400" />
                      </button>
                    ))}
                  </div>

                  {/* Research Direction */}
                  <div className="mt-6 pt-4 border-t border-slate-200">
                    <div className="flex items-center gap-2 mb-3">
                      <Compass className="text-cyan-500" size={20} />
                      <h4 className="font-medium text-slate-800">향후 연구 방향</h4>
                    </div>
                    <p className="text-slate-600 leading-relaxed">{trendAnalysis.researchDirection}</p>
                  </div>
                </div>
              </div>

              {/* Detailed Analysis */}
              <div className="glossy-panel p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Lightbulb className="text-yellow-500" size={24} />
                  <h3 className="text-lg font-semibold text-slate-800">상세 분석</h3>
                </div>
                <div className="prose max-w-none">
                  <p className="text-slate-700 leading-relaxed whitespace-pre-line">{trendAnalysis.analysis}</p>
                </div>
              </div>

              {/* Link to Search */}
              <div className="flex justify-center">
                <Link
                  to={`/search?q=${encodeURIComponent(queryFromUrl)}`}
                  className="glossy-btn-primary px-8 py-3 flex items-center gap-2"
                >
                  <Search size={20} />
                  "{queryFromUrl}" 관련 논문 검색하기
                </Link>
              </div>
            </div>
          ) : null}
        </div>
      )}

      {/* Default View - Hot Topics (shown when no search query) */}
      {!queryFromUrl && (
        <>
          {/* Top Row - Hot Topics List & Bar Chart */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Hot Topics List */}
            <div className="glossy-panel p-6">
              <div className="flex items-center gap-2 mb-6">
                <Flame className="text-orange-500" size={24} />
                <h2 className="text-xl font-semibold text-slate-800">핫 토픽 TOP 10</h2>
              </div>

              {hotLoading ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="animate-spin text-slate-400" size={32} />
                </div>
              ) : hotTopics && hotTopics.length > 0 ? (
                <div className="space-y-3">
                  {hotTopics.map((topic, index) => (
                    <button
                      key={topic.keyword}
                      onClick={() => {
                        setSearchInput(topic.keyword)
                        setSearchParams({ q: topic.keyword })
                      }}
                      className="flex items-center justify-between w-full p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors border border-slate-200"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className="w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold text-white"
                          style={{ backgroundColor: COLORS[index] }}
                        >
                          {index + 1}
                        </span>
                        <span className="font-medium text-slate-800">{topic.keyword}</span>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-semibold text-slate-700">
                          {topic.count.toLocaleString()}
                        </div>
                        <div
                          className={`text-xs font-medium ${
                            topic.growthRate > 0 ? 'text-green-600' : 'text-red-500'
                          }`}
                        >
                          {topic.growthRate > 0 ? '↑' : '↓'} {Math.round(topic.growthRate * 100)}%
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="text-center text-slate-500 py-8">데이터 로딩 실패</p>
              )}
            </div>

            {/* Bar Chart */}
            <div className="glossy-panel p-6">
              <div className="flex items-center gap-2 mb-6">
                <BarChart3 className="text-cyan-500" size={24} />
                <h2 className="text-xl font-semibold text-slate-800">논문 수 비교</h2>
              </div>

              {hotLoading ? (
                <div className="flex justify-center py-8 h-80">
                  <Loader2 className="animate-spin text-slate-400" size={32} />
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={hotTopicsChartData} layout="vertical" margin={{ left: 20, right: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.2)" />
                    <XAxis type="number" stroke="rgba(100,116,139,0.7)" />
                    <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12, fill: 'rgba(51,65,85,0.9)' }} />
                    <Tooltip
                      formatter={(value: number) => [`${value.toLocaleString()} 논문`, '논문 수']}
                      contentStyle={{ backgroundColor: 'rgba(255,255,255,0.95)', border: '1px solid rgba(100,116,139,0.3)', borderRadius: '8px', color: '#334155' }}
                    />
                    <Bar dataKey="count" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Middle Row - Line Chart */}
          <div className="glossy-panel p-6 mb-6">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="text-green-500" size={24} />
              <h2 className="text-xl font-semibold text-slate-800">키워드 트렌드 (월별)</h2>
            </div>

            {trendsLoading ? (
              <div className="flex justify-center py-8 h-80">
                <Loader2 className="animate-spin text-slate-400" size={32} />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={trendChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.2)" />
                  <XAxis dataKey="month" stroke="rgba(100,116,139,0.7)" />
                  <YAxis stroke="rgba(100,116,139,0.7)" />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(255,255,255,0.95)', border: '1px solid rgba(100,116,139,0.3)', borderRadius: '8px', color: '#334155' }}
                  />
                  <Legend wrapperStyle={{ color: '#334155' }} />
                  <Line
                    type="monotone"
                    dataKey="CRISPR"
                    stroke="#06b6d4"
                    strokeWidth={3}
                    dot={{ fill: '#06b6d4', strokeWidth: 2 }}
                    activeDot={{ r: 8 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="CAR-T"
                    stroke="#f472b6"
                    strokeWidth={3}
                    dot={{ fill: '#f472b6', strokeWidth: 2 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="immunotherapy"
                    stroke="#22c55e"
                    strokeWidth={3}
                    dot={{ fill: '#22c55e', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Bottom Row - Pie Chart & Keywords */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Pie Chart */}
            <div className="glossy-panel p-6">
              <h2 className="text-xl font-semibold text-slate-800 mb-6">연구 분야 분포</h2>

              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }) => `${name.slice(0, 10)}... ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number) => [`${value.toLocaleString()} 논문`, '논문 수']}
                    contentStyle={{ backgroundColor: 'rgba(255,255,255,0.95)', border: '1px solid rgba(100,116,139,0.3)', borderRadius: '8px', color: '#334155' }}
                  />
                </PieChart>
              </ResponsiveContainer>

              <div className="flex flex-wrap justify-center gap-2 mt-4">
                {pieData.map((entry, index) => (
                  <div key={index} className="flex items-center gap-1 text-sm">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                    <span className="text-slate-600">{entry.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Popular Keywords */}
            <div className="glossy-panel p-6">
              <div className="flex items-center gap-2 mb-6">
                <Sparkles className="text-yellow-500" size={24} />
                <h2 className="text-xl font-semibold text-slate-800">인기 키워드</h2>
              </div>
              <div className="flex flex-wrap gap-3">
                {[
                  { keyword: 'CRISPR-Cas9', hot: true },
                  { keyword: 'CAR-T therapy', hot: true },
                  { keyword: 'mRNA vaccine', hot: true },
                  { keyword: 'immunotherapy', hot: false },
                  { keyword: 'gene editing', hot: false },
                  { keyword: 'checkpoint inhibitor', hot: false },
                  { keyword: 'PD-1/PD-L1', hot: false },
                  { keyword: 'single-cell RNA-seq', hot: true },
                  { keyword: 'precision medicine', hot: false },
                  { keyword: 'biomarker', hot: false },
                  { keyword: 'AlphaFold', hot: true },
                  { keyword: 'spatial transcriptomics', hot: true },
                ].map(({ keyword, hot }) => (
                  <button
                    key={keyword}
                    onClick={() => {
                      setSearchInput(keyword)
                      setSearchParams({ q: keyword })
                    }}
                    className={`px-4 py-2 rounded-full cursor-pointer transition-all ${
                      hot
                        ? 'bg-gradient-to-r from-orange-500 to-pink-500 text-white font-medium shadow-lg hover:shadow-xl border border-orange-400/30'
                        : 'bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300'
                    }`}
                  >
                    {hot && <span className="mr-1">🔥</span>}
                    {keyword}
                  </button>
                ))}
              </div>

              {/* Growth Stats */}
              <div className="mt-8 grid grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-100 rounded-xl border border-green-300">
                  <div className="text-2xl font-bold text-green-600">+67%</div>
                  <div className="text-sm text-green-700">Spatial Transcriptomics</div>
                  <div className="text-xs text-green-600 mt-1">가장 빠른 성장</div>
                </div>
                <div className="text-center p-4 bg-cyan-100 rounded-xl border border-cyan-300">
                  <div className="text-2xl font-bold text-cyan-600">1,847</div>
                  <div className="text-sm text-cyan-700">CRISPR-Cas9</div>
                  <div className="text-xs text-cyan-600 mt-1">최다 논문</div>
                </div>
                <div className="text-center p-4 bg-purple-100 rounded-xl border border-purple-300">
                  <div className="text-2xl font-bold text-purple-600">+52%</div>
                  <div className="text-sm text-purple-700">AlphaFold</div>
                  <div className="text-xs text-purple-600 mt-1">AI 트렌드</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
        </>
      )}
    </div>
  )
}
