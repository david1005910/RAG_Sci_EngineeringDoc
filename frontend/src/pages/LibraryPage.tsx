import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Library, Tag, Trash2, ExternalLink, MessageSquare, Loader2, Search as SearchIcon, FolderOpen, FileDown, FileX, ChevronRight, ChevronDown, Code, FileText } from 'lucide-react'
import { Link } from 'react-router-dom'
import { libraryApi, vectordbApi, searchApi } from '@/services/api'
import { useAuthStore } from '@/store/authStore'
import type { PDFInfo } from '@/types'

export default function LibraryPage() {
  // Note: isAuthenticated can be used for future login check
  useAuthStore()  // Keep store subscription for future use
  const [selectedTag, setSelectedTag] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // 저장된 Meta 데이터 - vectordb_metadata.json에서 자동 저장된 전체 메타데이터
  const { data: metadataResponse, isLoading } = useQuery({
    queryKey: ['vectordbMetadata'],
    queryFn: () => vectordbApi.getMetadata(),
  })

  // Convert metadata to SavedPaper format for compatibility
  const papersData = metadataResponse?.papers?.map(p => ({
    id: p.pmid,
    pmid: p.pmid,
    title: p.title,
    abstract: p.abstract,
    authors: p.authors,
    journal: p.journal || '',
    tags: [],
    notes: undefined,
    saved_at: p.indexed_at
  }))

  // Tags are not used for auto-saved metadata (no user tags)
  const tags: string[] = []

  const deleteMutation = useMutation({
    mutationFn: (paperId: string) => libraryApi.deleteSavedPaper(paperId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['savedPapers'] })
      queryClient.invalidateQueries({ queryKey: ['tags'] })
    },
  })

  const handleDelete = (paperId: string, title: string) => {
    if (confirm(`"${title.substring(0, 50)}..." Meta 데이터를 삭제하시겠습니까?`)) {
      deleteMutation.mutate(paperId)
    }
  }

  // No login required for auto-saved metadata
  const showLoginPrompt = false

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold liquid-text">내 라이브러리</h1>
          <p className="liquid-text-muted mt-1">저장된 Meta 데이터를 관리하세요</p>
        </div>
        <div className="flex items-center gap-3">
          {/* 논문 검색 버튼 */}
          <Link
            to="/search"
            className="flex items-center gap-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 rounded-xl font-medium shadow-lg hover:shadow-xl transition-all hover:scale-105"
          >
            <SearchIcon size={20} />
            논문 검색
          </Link>
          {/* 통계 표시 */}
          <div className="flex items-center gap-2 bg-white/60 px-4 py-2 rounded-xl border border-slate-200">
            <FolderOpen className="text-purple-500" size={24} />
            <span className="text-slate-700 font-medium">
              {papersData?.length || 0}개의 Meta 데이터
            </span>
          </div>
        </div>
      </div>

      {/* Saved Papers Content */}
      {showLoginPrompt ? (
          <div className="glossy-panel p-12 text-center">
            <Library className="mx-auto text-white/30 mb-4" size={64} />
            <h2 className="text-2xl font-bold text-white mb-2">로그인이 필요합니다</h2>
            <p className="liquid-text-muted mb-6">
              논문을 저장하고 관리하려면 로그인하세요.
            </p>
            <Link
              to="/login"
              className="glossy-btn-primary inline-flex items-center px-6 py-3"
            >
              로그인
            </Link>
          </div>
        ) : (
        <div className="grid md:grid-cols-4 gap-6">
          {/* Sidebar - Tags */}
          <div className="md:col-span-1">
            <div className="glossy-panel p-4">
              <h3 className="font-medium text-slate-800 mb-3 flex items-center gap-2">
                <Tag size={16} className="text-cyan-500" />
                태그
              </h3>
              <div className="space-y-1">
                <button
                  onClick={() => setSelectedTag(null)}
                  className={`flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg transition-all ${
                    selectedTag === null
                      ? 'bg-cyan-100 text-cyan-700 border border-cyan-300'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  전체 보기
                </button>
                {tags && tags.length > 0 ? (
                  tags.map((tag) => (
                    <button
                      key={tag}
                      onClick={() => setSelectedTag(tag)}
                      className={`flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg transition-all ${
                        selectedTag === tag
                          ? 'bg-cyan-100 text-cyan-700 border border-cyan-300'
                          : 'text-slate-600 hover:bg-slate-100'
                      }`}
                    >
                      <Tag size={14} />
                      {tag}
                    </button>
                  ))
                ) : (
                  <p className="text-sm text-slate-500 px-3 py-2">태그가 없습니다</p>
                )}
              </div>
            </div>
          </div>

          {/* Papers List */}
          <div className="md:col-span-3">
            {isLoading ? (
              <div className="flex justify-center py-12">
                <div className="flex items-center gap-3 text-white/70">
                  <Loader2 className="animate-spin" size={24} />
                  <span>논문을 불러오는 중...</span>
                </div>
              </div>
            ) : papersData && papersData.length > 0 ? (
              <div className="glossy-panel divide-y divide-white/10">
                {papersData.map((paper, index) => (
                  <SavedPaperItem
                    key={paper.id}
                    paper={paper}
                    index={index + 1}
                    onDelete={handleDelete}
                    isDeleting={deleteMutation.isPending}
                  />
                ))}
              </div>
            ) : (
              <div className="glossy-panel p-12 text-center">
                <Library className="mx-auto text-white/30 mb-4" size={48} />
                <h3 className="text-lg font-medium text-white mb-2">
                  {selectedTag ? `"${selectedTag}" 태그의 Meta 데이터가 없습니다` : '저장된 Meta 데이터가 없습니다'}
                </h3>
                <p className="liquid-text-muted mb-6">
                  검색에서 논문을 찾아 저장해보세요
                </p>
                <Link
                  to="/search"
                  className="glossy-btn-primary inline-flex items-center gap-2 px-6 py-3"
                >
                  <SearchIcon size={18} />
                  논문 검색하기
                </Link>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// Saved Paper Item Component (목차 형태)
interface SavedPaper {
  id: string
  pmid: string
  title: string
  abstract: string
  authors?: string[]
  journal?: string
  tags: string[]
  notes?: string
  saved_at?: string
}

function SavedPaperItem({
  paper,
  index,
  onDelete,
  isDeleting
}: {
  paper: SavedPaper
  index: number
  onDelete: (id: string, title: string) => void
  isDeleting: boolean
}) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showJson, setShowJson] = useState(false)

  // JSON 형식의 메타데이터 생성
  const jsonMetadata = JSON.stringify({
    id: paper.id,
    pmid: paper.pmid,
    title: paper.title,
    abstract: paper.abstract,
    authors: paper.authors || [],
    journal: paper.journal || "",
    tags: paper.tags || [],
    notes: paper.notes || null,
    saved_at: paper.saved_at || null
  }, null, 2)

  return (
    <div className="transition-all bg-white/50 rounded-xl mb-2 border border-slate-200">
      {/* 제목 행 (목차) */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-cyan-50 transition-colors text-left rounded-xl"
      >
        {/* 번호 - 그라디언트 배지 */}
        <span className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 text-white font-bold text-sm flex items-center justify-center shadow-md">
          {index}
        </span>
        {isExpanded ? (
          <ChevronDown size={16} className="text-cyan-500 flex-shrink-0" />
        ) : (
          <ChevronRight size={16} className="text-slate-400 flex-shrink-0" />
        )}
        <span className="flex-1 text-slate-800 font-medium line-clamp-1">{paper.title}</span>
        {paper.tags && paper.tags.length > 0 && (
          <span className="px-2 py-0.5 bg-cyan-100 text-cyan-600 rounded text-xs">
            {paper.tags[0]}
          </span>
        )}
        <span className="text-xs text-slate-500 flex-shrink-0 bg-slate-100 px-2 py-1 rounded-lg">PMID: {paper.pmid}</span>
      </button>

      {/* 펼쳐진 상세 정보 */}
      {isExpanded && (
        <div className="px-4 pb-4 pt-1 ml-11 border-l-2 border-cyan-300">
          {/* 보기 모드 토글 */}
          <div className="flex items-center gap-2 mb-3">
            <button
              onClick={(e) => {
                e.stopPropagation()
                setShowJson(false)
              }}
              className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                !showJson
                  ? 'bg-cyan-100 text-cyan-700 border border-cyan-300'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <FileText size={12} />
              텍스트
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                setShowJson(true)
              }}
              className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                showJson
                  ? 'bg-green-100 text-green-700 border border-green-300'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <Code size={12} />
              JSON
            </button>
          </div>

          {showJson ? (
            /* JSON 형식 표시 */
            <div className="mb-3 p-3 bg-slate-800 rounded-lg border border-slate-600 overflow-x-auto">
              <pre className="text-xs text-green-400 font-mono whitespace-pre-wrap">
                {jsonMetadata}
              </pre>
            </div>
          ) : (
            /* 텍스트 형식 표시 */
            <>
              <div className="flex items-center gap-2 text-sm text-slate-600 mb-2">
                {paper.journal && <span>{paper.journal}</span>}
                <span className="px-2 py-0.5 bg-cyan-100 text-cyan-600 rounded text-xs font-medium">
                  <FolderOpen size={10} className="inline mr-1" />
                  Meta
                </span>
              </div>

              {paper.authors && paper.authors.length > 0 && (
                <div className="text-sm text-slate-500 mb-2">
                  저자: {paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ` 외 ${paper.authors.length - 3}명` : ''}
                </div>
              )}

              <p className="text-sm text-slate-600 mb-3 line-clamp-3">
                {paper.abstract}
              </p>

              {paper.tags && paper.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {paper.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 bg-cyan-100 text-cyan-700 rounded text-xs border border-cyan-300"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {paper.notes && (
                <div className="mb-3 p-2 bg-slate-100 rounded border border-slate-200">
                  <p className="text-xs text-slate-600">{paper.notes}</p>
                </div>
              )}
            </>
          )}

          <div className="flex items-center gap-3">
            <Link
              to={`/paper/${paper.pmid}`}
              className="flex items-center gap-1 text-sm text-cyan-600 hover:text-cyan-700 font-medium transition-colors"
            >
              <ExternalLink size={14} />
              상세보기
            </Link>
            <Link
              to={`/chat?pmid=${paper.pmid}`}
              className="flex items-center gap-1 text-sm text-pink-600 hover:text-pink-700 font-medium transition-colors"
            >
              <MessageSquare size={14} />
              AI 질문
            </Link>
            <PdfButton pmid={paper.pmid} />
            <button
              onClick={(e) => {
                e.stopPropagation()
                onDelete(paper.id, paper.title)
              }}
              disabled={isDeleting}
              className="flex items-center gap-1 text-sm text-red-500 hover:text-red-600 transition-colors ml-auto"
            >
              {isDeleting ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Trash2 size={14} />
              )}
              삭제
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// PDF Button Component
function PdfButton({ pmid }: { pmid: string }) {
  const [pdfInfo, setPdfInfo] = useState<PDFInfo | null>(null)
  const [isChecking, setIsChecking] = useState(false)

  const handleClick = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!pdfInfo) {
      setIsChecking(true)
      try {
        const info = await searchApi.getPdfInfo(pmid)
        setPdfInfo(info)
        if (info.hasPdf && info.pdfUrl) {
          window.open(info.pdfUrl, '_blank')
        }
      } catch (error) {
        console.error('PDF check failed:', error)
        setPdfInfo({ pmid, hasPdf: false, isOpenAccess: false })
      } finally {
        setIsChecking(false)
      }
    } else if (pdfInfo.hasPdf && pdfInfo.pdfUrl) {
      window.open(pdfInfo.pdfUrl, '_blank')
    }
  }

  return (
    <button
      onClick={handleClick}
      disabled={isChecking || (pdfInfo !== null && !pdfInfo.hasPdf)}
      className={`flex items-center gap-1 text-sm transition-colors ${
        pdfInfo?.hasPdf
          ? 'text-green-300 hover:text-green-200'
          : pdfInfo !== null
            ? 'text-white/30 cursor-not-allowed'
            : 'text-white/60 hover:text-green-300'
      }`}
      title={
        pdfInfo?.hasPdf
          ? 'PDF 다운로드'
          : pdfInfo !== null
            ? 'PDF 미제공'
            : 'PDF 확인'
      }
    >
      {isChecking ? (
        <Loader2 size={14} className="animate-spin" />
      ) : pdfInfo?.hasPdf ? (
        <FileDown size={14} />
      ) : pdfInfo !== null ? (
        <FileX size={14} />
      ) : (
        <FileDown size={14} />
      )}
      {isChecking ? '확인 중' : pdfInfo?.hasPdf ? 'PDF' : pdfInfo !== null ? '미제공' : 'PDF'}
    </button>
  )
}
