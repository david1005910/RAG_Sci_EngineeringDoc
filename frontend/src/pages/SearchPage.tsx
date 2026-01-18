import React, { useState, useEffect, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Search, Filter, ExternalLink, MessageSquare, Bookmark, BookmarkCheck, Languages, Loader2, Database, CheckCircle, X, Calendar, BookOpen, Users, TrendingUp, ChevronLeft, ChevronRight, FileDown, FileX, Globe, Beaker, GraduationCap, Library, ChevronDown, ChevronUp } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { searchApi, libraryApi, vectordbApi } from '@/services/api'
import { searchArxiv, searchCrossRef, type ExternalPaper, type ExternalSearchResponse } from '@/services/externalApis'
import { useAuthStore } from '@/store/authStore'
import { useSearchStore } from '@/store/searchStore'
import { validateSearchQuery, validateYearParam, validateStringArrayParam } from '@/utils/validation'
import { logError, getErrorMessage } from '@/utils/errorHandler'
import type { PaperSearchResult, PDFInfo } from '@/types'

// Simple Korean detection
function containsKorean(text: string): boolean {
  return /[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]/.test(text)
}

// Filter types
interface SearchFilters {
  yearFrom?: number
  yearTo?: number
  journals?: string[]
  authors?: string[]
}

// Parse filters from URL params with validation
function parseFiltersFromUrl(searchParams: URLSearchParams): SearchFilters {
  const filters: SearchFilters = {}

  // Validate year parameters
  filters.yearFrom = validateYearParam(searchParams.get('yearFrom'))
  filters.yearTo = validateYearParam(searchParams.get('yearTo'))

  // Validate and sanitize string arrays
  filters.journals = validateStringArrayParam(searchParams.get('journals'), 10, 100)
  filters.authors = validateStringArrayParam(searchParams.get('authors'), 20, 100)

  // Validate year range
  if (filters.yearFrom && filters.yearTo && filters.yearFrom > filters.yearTo) {
    filters.yearFrom = undefined
    filters.yearTo = undefined
  }

  return filters
}

// Convert filters to URL params
function filtersToUrlParams(filters: SearchFilters): Record<string, string> {
  const params: Record<string, string> = {}
  if (filters.yearFrom) params.yearFrom = filters.yearFrom.toString()
  if (filters.yearTo) params.yearTo = filters.yearTo.toString()
  if (filters.journals?.length) params.journals = filters.journals.join(',')
  if (filters.authors?.length) params.authors = filters.authors.join(',')
  return params
}

// Constants for pagination
const PAPERS_PER_PAGE = 10
const TOTAL_FETCH_LIMIT = 50

export default function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams()

  // Zustand store for persisting search state across navigation
  const {
    lastQuery,
    lastFilters,
    currentPage: storedPage,
    autoSavedQuery: storedAutoSavedQuery,
    setLastSearch,
    setCurrentPage: setStoredPage,
    setAutoSavedQuery: setStoredAutoSavedQuery,
  } = useSearchStore()

  // Initialize from URL params, or fall back to stored state
  const urlQuery = searchParams.get('q')
  const initialQuery = urlQuery || lastQuery || ''

  const [query, setQuery] = useState(initialQuery)
  const [isSavingToVectorDB, setIsSavingToVectorDB] = useState(false)
  const [vectorDBSaveResult, setVectorDBSaveResult] = useState<{ saved: number; chunks: number } | null>(null)
  const [autoSavedQuery, setAutoSavedQuery] = useState<string | null>(storedAutoSavedQuery)

  // Pagination state - initialize from URL or store
  const [currentPage, setCurrentPage] = useState(() => {
    const pageParam = searchParams.get('page')
    return pageParam ? parseInt(pageParam) : storedPage || 1
  })

  // Filter state - initialize from URL or store
  const [showFilters, setShowFilters] = useState(false)
  const urlFilters = parseFiltersFromUrl(searchParams)
  const hasUrlFilters = Object.keys(urlFilters).length > 0
  const initialFilters = hasUrlFilters ? urlFilters : lastFilters || {}

  const [filters, setFilters] = useState<SearchFilters>(initialFilters)
  const [tempFilters, setTempFilters] = useState<SearchFilters>(initialFilters)
  const [journalInput, setJournalInput] = useState('')
  const [authorInput, setAuthorInput] = useState('')

  // Restore URL from store if URL has no query but store has one
  useEffect(() => {
    if (!urlQuery && lastQuery) {
      const filterParams = filtersToUrlParams(lastFilters)
      const newParams: Record<string, string> = { q: lastQuery, ...filterParams }
      if (storedPage > 1) {
        newParams.page = storedPage.toString()
      }
      setSearchParams(newParams, { replace: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Run only on mount

  // Get search term - use URL first, then stored
  const searchTerm = urlQuery || lastQuery || ''
  const isKoreanSearch = containsKorean(searchTerm)

  // Use React Query for translation (with caching)
  const { data: translationData, isLoading: isTranslating } = useQuery({
    queryKey: ['translate', searchTerm],
    queryFn: async () => {
      const result = await searchApi.translate(searchTerm, 'ko', 'en')
      return result.translated
    },
    enabled: !!searchTerm && isKoreanSearch,
    staleTime: 60 * 60 * 1000, // Cache translation for 1 hour
    gcTime: 2 * 60 * 60 * 1000, // Keep in cache for 2 hours
    refetchOnMount: false, // Don't refetch when navigating back
    refetchOnWindowFocus: false, // Don't refetch on window focus
  })

  const translatedQuery = translationData || ''

  // Sync query input with URL
  useEffect(() => {
    const urlQuery = searchParams.get('q')
    if (urlQuery && urlQuery !== query) {
      setQuery(urlQuery)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

  // Sync filters and page from URL when navigating back
  useEffect(() => {
    const urlFilters = parseFiltersFromUrl(searchParams)
    const hasUrlFilters = Object.keys(urlFilters).length > 0
    if (hasUrlFilters) {
      setFilters(urlFilters)
      setTempFilters(urlFilters)
    }

    // Sync page from URL
    const pageParam = searchParams.get('page')
    if (pageParam) {
      const page = parseInt(pageParam)
      if (page !== currentPage && page >= 1) {
        setCurrentPage(page)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

  // Convert filters for API
  const apiFilters = Object.keys(filters).length > 0 ? {
    year_from: filters.yearFrom,
    year_to: filters.yearTo,
    journals: filters.journals,
    authors: filters.authors,
  } : undefined

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', isKoreanSearch ? translatedQuery : searchTerm, filters],
    queryFn: () => searchApi.search(isKoreanSearch ? translatedQuery : searchTerm, TOTAL_FETCH_LIMIT, apiFilters),
    enabled: !!(isKoreanSearch ? translatedQuery : searchTerm) && !isTranslating,
    staleTime: 30 * 60 * 1000, // Cache for 30 minutes (increased)
    gcTime: 60 * 60 * 1000, // Keep in cache for 1 hour (increased)
    refetchOnMount: false, // Don't refetch when navigating back
    refetchOnWindowFocus: false, // Don't refetch on window focus
  })

  // Auto-save search results to VectorDB
  useEffect(() => {
    const currentQuery = isKoreanSearch ? translatedQuery : searchTerm

    // Skip if no data, already saving, already saved this query, or no results
    if (!data?.results?.length || isSavingToVectorDB || autoSavedQuery === currentQuery) {
      return
    }

    const autoSaveToVectorDB = async () => {
      setIsSavingToVectorDB(true)
      try {
        const papers = data.results.map((paper: PaperSearchResult) => ({
          pmid: paper.pmid,
          title: paper.title,
          abstract: paper.abstract,
          authors: paper.authors,
          journal: paper.journal,
          publication_date: paper.publicationDate,
          keywords: paper.keywords || [],
        }))

        const result = await vectordbApi.savePapers(papers)
        setVectorDBSaveResult({
          saved: result.saved_count,
          chunks: result.total_chunks,
        })
        setAutoSavedQuery(currentQuery) // Mark this query as saved
        console.log(`Auto-saved ${result.saved_count} papers to VectorDB`)
      } catch (error) {
        logError('Auto-save to VectorDB', error)
      } finally {
        setIsSavingToVectorDB(false)
      }
    }

    autoSaveToVectorDB()
  }, [data, searchTerm, translatedQuery, isKoreanSearch, isSavingToVectorDB, autoSavedQuery])

  // Save search state to store when results arrive (for persistence across navigation)
  useEffect(() => {
    if (data?.results && searchTerm) {
      setLastSearch(searchTerm, data.results, filters)
    }
  }, [data, searchTerm, filters, setLastSearch])

  // Sync autoSavedQuery with store
  useEffect(() => {
    if (autoSavedQuery !== storedAutoSavedQuery) {
      setStoredAutoSavedQuery(autoSavedQuery)
    }
  }, [autoSavedQuery, storedAutoSavedQuery, setStoredAutoSavedQuery])

  // Sync current page with store
  useEffect(() => {
    if (currentPage !== storedPage) {
      setStoredPage(currentPage)
    }
  }, [currentPage, storedPage, setStoredPage])

  // Calculate pagination
  const totalResults = data?.results?.length || 0
  const totalPages = Math.ceil(totalResults / PAPERS_PER_PAGE)
  const startIndex = (currentPage - 1) * PAPERS_PER_PAGE
  const endIndex = startIndex + PAPERS_PER_PAGE
  const currentPageResults = data?.results?.slice(startIndex, endIndex) || []

  // Update URL when page changes (but not on initial load)
  const updatePageInUrl = (page: number) => {
    const filterParams = filtersToUrlParams(filters)
    if (page === 1) {
      // Don't include page=1 in URL (it's the default)
      setSearchParams({ q: searchTerm, ...filterParams })
    } else {
      setSearchParams({ q: searchTerm, ...filterParams, page: page.toString() })
    }
  }

  // Handle page change with URL update
  const handlePageChange = (page: number) => {
    setCurrentPage(page)
    updatePageInUrl(page)
    // Scroll to top of results
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Reset pagination when search term or filters change (not when navigating back)
  const prevSearchTermRef = useRef(searchTerm)
  const prevFiltersRef = useRef(filters)

  useEffect(() => {
    // Only reset if search term or filters actually changed (not from URL sync)
    if (prevSearchTermRef.current !== searchTerm ||
        JSON.stringify(prevFiltersRef.current) !== JSON.stringify(filters)) {
      // Check if this is from URL sync (page already in URL)
      const pageParam = searchParams.get('page')
      if (!pageParam) {
        setCurrentPage(1)
      }
      prevSearchTermRef.current = searchTerm
      prevFiltersRef.current = filters
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchTerm, filters])

  // Check if any filters are active
  const hasActiveFilters = filters.yearFrom || filters.yearTo ||
    (filters.journals && filters.journals.length > 0) ||
    (filters.authors && filters.authors.length > 0)

  // Apply filters and update URL
  const handleApplyFilters = () => {
    setFilters(tempFilters)
    setShowFilters(false)
    // Update URL with filters
    if (searchTerm) {
      const filterParams = filtersToUrlParams(tempFilters)
      setSearchParams({ q: searchTerm, ...filterParams })
    }
  }

  // Clear all filters and update URL
  const handleClearFilters = () => {
    setFilters({})
    setTempFilters({})
    setJournalInput('')
    setAuthorInput('')
    // Update URL without filters
    if (searchTerm) {
      setSearchParams({ q: searchTerm })
    }
  }

  // Add journal to filter
  const handleAddJournal = () => {
    if (journalInput.trim()) {
      setTempFilters(prev => ({
        ...prev,
        journals: [...(prev.journals || []), journalInput.trim()]
      }))
      setJournalInput('')
    }
  }

  // Remove journal from filter
  const handleRemoveJournal = (journal: string) => {
    setTempFilters(prev => ({
      ...prev,
      journals: prev.journals?.filter(j => j !== journal)
    }))
  }

  // Add author to filter
  const handleAddAuthor = () => {
    if (authorInput.trim()) {
      setTempFilters(prev => ({
        ...prev,
        authors: [...(prev.authors || []), authorInput.trim()]
      }))
      setAuthorInput('')
    }
  }

  // Remove author from filter
  const handleRemoveAuthor = (author: string) => {
    setTempFilters(prev => ({
      ...prev,
      authors: prev.authors?.filter(a => a !== author)
    }))
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()

    // Validate and sanitize search query
    const validation = validateSearchQuery(query)
    if (!validation.valid || !validation.sanitized) {
      return
    }

    // Reset VectorDB save result for new search
    setVectorDBSaveResult(null)

    // Update URL params with validated query and current filters
    const filterParams = filtersToUrlParams(filters)
    setSearchParams({ q: validation.sanitized, ...filterParams })
  }

  // Save all search results to VectorDB
  const handleSaveToVectorDB = async () => {
    if (!data?.results?.length) return

    setIsSavingToVectorDB(true)
    setVectorDBSaveResult(null)

    try {
      const papers = data.results.map((paper: PaperSearchResult) => ({
        pmid: paper.pmid,
        title: paper.title,
        abstract: paper.abstract,
        authors: paper.authors,
        journal: paper.journal,
        publication_date: paper.publicationDate,
        keywords: paper.keywords || [],
      }))

      const result = await vectordbApi.savePapers(papers)
      setVectorDBSaveResult({
        saved: result.saved_count,
        chunks: result.total_chunks,
      })
    } catch (error) {
      logError('VectorDB save', error)
      alert(getErrorMessage(error))
    } finally {
      setIsSavingToVectorDB(false)
    }
  }

  // VectorDB indexed papers query
  const { data: vectordbPapers } = useQuery({
    queryKey: ['vectordbPapers'],
    queryFn: () => vectordbApi.getPapers(),
  })

  const [showVectordbPanel, setShowVectordbPanel] = useState(true)

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold liquid-text mb-2">논문 검색</h1>
        <p className="liquid-text-muted">
          한글 또는 영어로 검색하면 AI가 의미를 이해하고 관련 논문을 찾아드립니다.
        </p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-white/50" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="예: 암 면역치료 부작용, CRISPR gene therapy..."
              className="glossy-input w-full pl-12 pr-4 py-4"
            />
          </div>
          <button
            type="submit"
            disabled={!query.trim() || isLoading || isTranslating}
            className="glossy-btn-primary px-8 py-4 font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading || isTranslating ? <Loader2 className="animate-spin" size={20} /> : <Search size={20} />}
            {isTranslating ? '번역 중...' : '검색'}
          </button>
        </div>
      </form>

      {/* Translation Notice */}
      {isKoreanSearch && translatedQuery && searchTerm && (
        <div className="glossy-panel-sm bg-cyan-500/20 p-4 mb-6 flex items-center gap-3">
          <Languages className="text-cyan-300" size={20} />
          <div>
            <span className="text-white/70">한글 검색: </span>
            <span className="text-white font-medium">"{searchTerm}"</span>
            <span className="text-white/50 mx-2">→</span>
            <span className="text-white/70">영어 변환: </span>
            <span className="text-cyan-300 font-medium">"{translatedQuery}"</span>
          </div>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="flex gap-6">
        {/* Left: Search Results */}
        <div className="flex-1 min-w-0">
          {/* Results */}
          {(isLoading || isTranslating) && (
            <div className="flex justify-center py-12">
              <div className="flex items-center gap-3 text-white/70">
                <Loader2 className="animate-spin" size={24} />
                <span>{isTranslating ? 'AI 번역 중...' : 'PubMed에서 검색 중...'}</span>
              </div>
            </div>
          )}

          {error && (
            <div className="glossy-panel-sm bg-red-500/20 text-white p-4">
              검색 중 오류가 발생했습니다. 다시 시도해주세요.
            </div>
          )}

      {data && (
        <div>
          <div className="flex items-center justify-between mb-6">
            <p className="liquid-text-muted">
              <span className="font-semibold text-white">{data.total.toLocaleString()}</span>건의 결과
              <span className="text-white/50 ml-2">({data.tookMs}ms)</span>
            </p>
            <div className="flex items-center gap-3">
              {/* VectorDB Indexing Button */}
              {data.results.length > 0 && (
                <button
                  onClick={handleSaveToVectorDB}
                  disabled={isSavingToVectorDB || !!vectorDBSaveResult}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    vectorDBSaveResult
                      ? 'bg-green-500/30 text-green-200 border border-green-400/30'
                      : 'glossy-btn-primary hover:scale-105'
                  }`}
                  title={`전체 ${totalResults}개 논문을 벡터화하여 VectorDB에 저장`}
                >
                  {isSavingToVectorDB ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      <span>{totalResults}개 논문 인덱싱 중...</span>
                    </>
                  ) : vectorDBSaveResult ? (
                    <>
                      <CheckCircle size={18} />
                      <span>완료: {vectorDBSaveResult.saved}개 논문 ({vectorDBSaveResult.chunks} chunks)</span>
                    </>
                  ) : (
                    <>
                      <Database size={18} />
                      <span>전체 {totalResults}개 논문 인덱싱</span>
                    </>
                  )}
                </button>
              )}
              <button
                onClick={() => {
                  setTempFilters(filters)
                  setShowFilters(!showFilters)
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  hasActiveFilters
                    ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
                    : 'glossy-btn'
                }`}
              >
                <Filter size={18} />
                필터
                {hasActiveFilters && (
                  <span className="px-1.5 py-0.5 bg-cyan-500/50 rounded-full text-xs">
                    {(filters.yearFrom || filters.yearTo ? 1 : 0) +
                      (filters.journals?.length || 0) +
                      (filters.authors?.length || 0)}
                  </span>
                )}
              </button>
              {/* Trend Analysis Button */}
              <Link
                to={`/trends?q=${encodeURIComponent(searchTerm)}`}
                className="flex items-center gap-2 px-4 py-2 rounded-lg glossy-btn hover:scale-105 transition-all"
                title="이 검색어에 대한 연구 트렌드 분석"
              >
                <TrendingUp size={18} />
                트렌드 분석
              </Link>
            </div>
          </div>

          {/* Filter Panel */}
          {showFilters && (
            <div className="glossy-panel p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Filter size={20} className="text-cyan-300" />
                  검색 필터
                </h3>
                <button
                  onClick={() => setShowFilters(false)}
                  className="p-1 hover:bg-white/10 rounded"
                >
                  <X size={20} className="text-white/70" />
                </button>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Year Range */}
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-white/80 mb-2">
                    <Calendar size={16} className="text-cyan-300" />
                    출판 연도
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      placeholder="시작 년도"
                      value={tempFilters.yearFrom || ''}
                      onChange={(e) => setTempFilters(prev => ({
                        ...prev,
                        yearFrom: e.target.value ? parseInt(e.target.value) : undefined
                      }))}
                      className="glossy-input px-3 py-2 w-full"
                      min={1900}
                      max={new Date().getFullYear()}
                    />
                    <span className="text-white/50">~</span>
                    <input
                      type="number"
                      placeholder="종료 년도"
                      value={tempFilters.yearTo || ''}
                      onChange={(e) => setTempFilters(prev => ({
                        ...prev,
                        yearTo: e.target.value ? parseInt(e.target.value) : undefined
                      }))}
                      className="glossy-input px-3 py-2 w-full"
                      min={1900}
                      max={new Date().getFullYear()}
                    />
                  </div>
                </div>

                {/* Journals */}
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-white/80 mb-2">
                    <BookOpen size={16} className="text-cyan-300" />
                    저널명
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="저널명 입력"
                      value={journalInput}
                      onChange={(e) => setJournalInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddJournal())}
                      className="glossy-input px-3 py-2 flex-1"
                    />
                    <button
                      onClick={handleAddJournal}
                      className="glossy-btn px-3 py-2"
                    >
                      추가
                    </button>
                  </div>
                  {tempFilters.journals && tempFilters.journals.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {tempFilters.journals.map((journal) => (
                        <span
                          key={journal}
                          className="flex items-center gap-1 px-2 py-1 bg-cyan-500/20 text-cyan-200 rounded-full text-sm"
                        >
                          {journal}
                          <button
                            onClick={() => handleRemoveJournal(journal)}
                            className="hover:bg-white/20 rounded-full p-0.5"
                          >
                            <X size={12} />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Authors */}
                <div className="md:col-span-2">
                  <label className="flex items-center gap-2 text-sm font-medium text-white/80 mb-2">
                    <Users size={16} className="text-cyan-300" />
                    저자명
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="저자명 입력 (예: Kim, Smith)"
                      value={authorInput}
                      onChange={(e) => setAuthorInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddAuthor())}
                      className="glossy-input px-3 py-2 flex-1"
                    />
                    <button
                      onClick={handleAddAuthor}
                      className="glossy-btn px-3 py-2"
                    >
                      추가
                    </button>
                  </div>
                  {tempFilters.authors && tempFilters.authors.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {tempFilters.authors.map((author) => (
                        <span
                          key={author}
                          className="flex items-center gap-1 px-2 py-1 bg-green-500/20 text-green-200 rounded-full text-sm"
                        >
                          {author}
                          <button
                            onClick={() => handleRemoveAuthor(author)}
                            className="hover:bg-white/20 rounded-full p-0.5"
                          >
                            <X size={12} />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Filter Actions */}
              <div className="flex items-center justify-end gap-3 mt-6 pt-4 border-t border-white/10">
                <button
                  onClick={handleClearFilters}
                  className="glossy-btn px-4 py-2 text-white/70"
                >
                  초기화
                </button>
                <button
                  onClick={handleApplyFilters}
                  className="glossy-btn-primary px-6 py-2"
                >
                  필터 적용
                </button>
              </div>
            </div>
          )}

          {/* Active Filters Display */}
          {hasActiveFilters && !showFilters && (
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <span className="text-sm text-white/60">적용된 필터:</span>
              {(filters.yearFrom || filters.yearTo) && (
                <span className="px-2 py-1 bg-cyan-500/20 text-cyan-200 rounded-full text-sm flex items-center gap-1">
                  <Calendar size={12} />
                  {filters.yearFrom || '?'} ~ {filters.yearTo || '?'}
                </span>
              )}
              {filters.journals?.map((j) => (
                <span key={j} className="px-2 py-1 bg-cyan-500/20 text-cyan-200 rounded-full text-sm flex items-center gap-1">
                  <BookOpen size={12} />
                  {j}
                </span>
              ))}
              {filters.authors?.map((a) => (
                <span key={a} className="px-2 py-1 bg-green-500/20 text-green-200 rounded-full text-sm flex items-center gap-1">
                  <Users size={12} />
                  {a}
                </span>
              ))}
              <button
                onClick={handleClearFilters}
                className="text-sm text-red-300 hover:text-red-200 flex items-center gap-1"
              >
                <X size={14} />
                필터 초기화
              </button>
            </div>
          )}

          <div className="space-y-4">
            {currentPageResults.map((paper: PaperSearchResult) => (
              <PaperCard key={paper.pmid} paper={paper} />
            ))}
          </div>

          {data.results.length === 0 && (
            <div className="text-center py-12 liquid-text-muted">
              검색 결과가 없습니다. 다른 키워드로 검색해보세요.
            </div>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-4 mt-8 pt-6 border-t border-white/10">
              <button
                onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="flex items-center gap-1 px-4 py-2 glossy-btn disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft size={18} />
                이전
              </button>

              <div className="flex items-center gap-2">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                  <button
                    key={page}
                    onClick={() => handlePageChange(page)}
                    className={`w-10 h-10 rounded-lg font-medium transition-all ${
                      currentPage === page
                        ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
                        : 'glossy-btn hover:scale-105'
                    }`}
                  >
                    {page}
                  </button>
                ))}
              </div>

              <button
                onClick={() => handlePageChange(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="flex items-center gap-1 px-4 py-2 glossy-btn disabled:opacity-50 disabled:cursor-not-allowed"
              >
                다음
                <ChevronRight size={18} />
              </button>
            </div>
          )}

          {/* Info about total papers fetched */}
          <div className="text-center mt-4 text-sm text-white/50">
            총 {totalResults}개 논문 중 {startIndex + 1}-{Math.min(endIndex, totalResults)}번째 표시
          </div>
        </div>
      )}

      {/* Empty state */}
      {!searchTerm && (
        <div className="text-center py-16">
          <Search className="mx-auto text-purple-300 mb-4" size={64} />
          <h3 className="text-xl font-medium liquid-text mb-2">
            검색어를 입력하세요
          </h3>
          <p className="liquid-text-muted max-w-md mx-auto mb-6">
            한글 또는 영어로 검색할 수 있습니다.
          </p>
          <div className="flex flex-wrap gap-2 justify-center max-w-2xl mx-auto">
            {[
              '암 면역치료 최신 동향',
              'CRISPR gene therapy',
              '유전자 편집 부작용',
              'CAR-T cell therapy',
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => {
                  setQuery(suggestion)
                  setSearchParams({ q: suggestion })
                }}
                className="glossy-btn px-4 py-2 text-sm hover:scale-105 transition-all"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

          {/* External Sources Section */}
          {searchTerm && (
            <ExternalSourcesSection query={isKoreanSearch ? translatedQuery : searchTerm} />
          )}
        </div>

        {/* Right: VectorDB Indexed Papers Sidebar */}
        <div className="w-80 flex-shrink-0 hidden lg:block">
          <div className="glossy-panel p-4 sticky top-24 max-h-[calc(100vh-8rem)] flex flex-col">
            {/* Header */}
            <button
              onClick={() => setShowVectordbPanel(!showVectordbPanel)}
              className="w-full flex items-center justify-between mb-3 flex-shrink-0"
            >
              <div className="flex items-center gap-2">
                <Database className="text-purple-500" size={20} />
                <h3 className="font-semibold text-slate-800">VectorDB 인덱싱 논문</h3>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs bg-purple-100 text-purple-600 px-2 py-0.5 rounded-full font-medium">
                  {vectordbPapers?.total || 0}
                </span>
                {showVectordbPanel ? (
                  <ChevronUp size={18} className="text-slate-400" />
                ) : (
                  <ChevronDown size={18} className="text-slate-400" />
                )}
              </div>
            </button>

            {/* Paper List */}
            {showVectordbPanel && (
              <div className="space-y-2 flex-1 overflow-y-auto min-h-0">
                {vectordbPapers && vectordbPapers.papers.length > 0 ? (
                  vectordbPapers.papers.slice(0, 20).map((paper, index) => (
                    <Link
                      key={paper.id}
                      to={`/paper/${paper.pmid}`}
                      className="flex items-start gap-2 p-2 bg-white/50 rounded-lg border border-slate-200 hover:bg-purple-50 transition-colors group"
                    >
                      <span className="flex-shrink-0 w-6 h-6 rounded bg-gradient-to-br from-purple-500 to-pink-500 text-white text-xs font-bold flex items-center justify-center">
                        {index + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-slate-700 font-medium line-clamp-2 group-hover:text-purple-700">
                          {paper.title}
                        </p>
                        <p className="text-xs text-slate-500 mt-1">PMID: {paper.pmid}</p>
                      </div>
                    </Link>
                  ))
                ) : (
                  <div className="text-center py-6 text-slate-500">
                    <Database className="mx-auto mb-2 opacity-30" size={32} />
                    <p className="text-sm">인덱싱된 논문이 없습니다</p>
                  </div>
                )}

                {/* View All Link */}
                {vectordbPapers && vectordbPapers.papers.length > 20 && (
                  <Link
                    to="/library"
                    className="flex items-center justify-center gap-2 p-2 text-sm text-purple-600 hover:text-purple-700 font-medium"
                  >
                    <Library size={16} />
                    전체 {vectordbPapers.total}개 보기
                  </Link>
                )}
              </div>
            )}

            {/* Library Link */}
            <Link
              to="/library"
              className="flex items-center justify-center gap-2 mt-3 pt-3 border-t border-slate-200 text-sm text-purple-600 hover:text-purple-700 font-medium flex-shrink-0"
            >
              <Library size={16} />
              내 라이브러리에서 관리
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}

// External Sources Section Component
function ExternalSourcesSection({ query }: { query: string }) {
  const [activeSource, setActiveSource] = useState<'arxiv' | 'crossref' | null>(null)
  const [results, setResults] = useState<ExternalSearchResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const searchExternalSource = async (source: 'arxiv' | 'crossref') => {
    if (activeSource === source && results) {
      // Toggle off
      setActiveSource(null)
      setResults(null)
      return
    }

    setIsLoading(true)
    setError(null)
    setActiveSource(source)

    try {
      const response = source === 'arxiv'
        ? await searchArxiv(query, 10)
        : await searchCrossRef(query, 10)
      setResults(response)
    } catch (err) {
      setError(`${source} 검색 중 오류가 발생했습니다.`)
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="mt-8 glossy-panel p-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
        <Globe className="text-purple-500" size={20} />
        외부 논문 데이터베이스 검색
      </h3>
      <p className="text-sm text-slate-600 mb-4">
        과학/공학 분야 논문을 외부 데이터베이스에서 검색합니다.
      </p>

      {/* Source buttons */}
      <div className="flex flex-wrap gap-3 mb-6">
        <button
          onClick={() => searchExternalSource('arxiv')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
            activeSource === 'arxiv'
              ? 'bg-gradient-to-r from-red-400 to-orange-400 text-white shadow-lg'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
          }`}
        >
          <Beaker size={18} />
          arXiv
          <span className="text-xs opacity-75">(물리/수학/CS)</span>
        </button>

        <button
          onClick={() => searchExternalSource('crossref')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
            activeSource === 'crossref'
              ? 'bg-gradient-to-r from-blue-400 to-cyan-400 text-white shadow-lg'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
          }`}
        >
          <GraduationCap size={18} />
          CrossRef
          <span className="text-xs opacity-75">(전 분야)</span>
        </button>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="animate-spin text-purple-500 mr-2" size={24} />
          <span className="text-slate-600">검색 중...</span>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="bg-red-50 text-red-600 p-4 rounded-xl mb-4">
          {error}
        </div>
      )}

      {/* Results */}
      {results && !isLoading && (
        <div className="space-y-4">
          <div className="text-sm text-slate-600 mb-2">
            <span className="font-medium text-purple-600">{results.source}</span>에서{' '}
            <span className="font-semibold">{results.total.toLocaleString()}</span>건 중{' '}
            <span className="font-semibold">{results.papers.length}</span>건 표시
          </div>

          {results.papers.length === 0 ? (
            <div className="text-center py-8 text-slate-500">
              검색 결과가 없습니다.
            </div>
          ) : (
            <div className="space-y-3">
              {results.papers.map((paper) => (
                <ExternalPaperCard key={paper.id} paper={paper} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Info when no source selected */}
      {!activeSource && !isLoading && (
        <div className="text-center py-6 text-slate-500">
          <Globe className="mx-auto mb-2 opacity-30" size={40} />
          <p>위 버튼을 클릭하여 외부 데이터베이스를 검색하세요</p>
        </div>
      )}
    </div>
  )
}

// External Paper Card Component
function ExternalPaperCard({ paper }: { paper: ExternalPaper }) {
  const [showAbstract, setShowAbstract] = useState(false)

  const sourceColors = {
    arxiv: 'bg-gradient-to-r from-red-100 to-orange-100 text-red-700',
    crossref: 'bg-gradient-to-r from-blue-100 to-cyan-100 text-blue-700',
    semanticscholar: 'bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700',
  }

  const sourceLabels = {
    arxiv: 'arXiv',
    crossref: 'CrossRef',
    semanticscholar: 'Semantic Scholar',
  }

  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex-1">
          <h4 className="font-medium text-slate-800 leading-snug">
            {paper.title}
          </h4>
        </div>
        <span className={`text-xs px-2 py-1 rounded-lg font-medium ${sourceColors[paper.source]}`}>
          {sourceLabels[paper.source]}
        </span>
      </div>

      {/* Authors and Year */}
      <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600 mb-2">
        {paper.authors.length > 0 && (
          <span className="flex items-center gap-1">
            <Users size={14} />
            {paper.authors.slice(0, 3).join(', ')}
            {paper.authors.length > 3 && ` 외 ${paper.authors.length - 3}명`}
          </span>
        )}
        {paper.year && (
          <span className="flex items-center gap-1">
            <Calendar size={14} />
            {paper.year}
          </span>
        )}
      </div>

      {/* Categories (for arXiv) */}
      {paper.categories && paper.categories.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {paper.categories.slice(0, 3).map((cat) => (
            <span key={cat} className="text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded">
              {cat}
            </span>
          ))}
        </div>
      )}

      {/* Abstract toggle */}
      {paper.abstract && (
        <div className="mb-2">
          <button
            onClick={() => setShowAbstract(!showAbstract)}
            className="text-sm text-purple-600 hover:text-purple-700 font-medium"
          >
            {showAbstract ? '초록 숨기기 ▲' : '초록 보기 ▼'}
          </button>
          {showAbstract && (
            <p className="mt-2 text-sm text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-lg">
              {paper.abstract.length > 500
                ? `${paper.abstract.substring(0, 500)}...`
                : paper.abstract}
            </p>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3 mt-3 pt-3 border-t border-slate-100">
        <a
          href={paper.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-sm text-purple-600 hover:text-purple-700 font-medium"
        >
          <ExternalLink size={14} />
          논문 보기
        </a>
        {paper.doi && (
          <a
            href={`https://doi.org/${paper.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            <BookOpen size={14} />
            DOI
          </a>
        )}
      </div>
    </div>
  )
}

interface PaperCardProps {
  paper: PaperSearchResult
}

function PaperCard({ paper }: PaperCardProps) {
  const relevancePercent = Math.round(paper.relevanceScore * 100)
  const [koreanSummary, setKoreanSummary] = useState<string | null>(null)
  const [isLoadingSummary, setIsLoadingSummary] = useState(false)
  const [summaryError, setSummaryError] = useState(false)
  const [showKorean, setShowKorean] = useState(false)
  const [isSaved, setIsSaved] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [pdfInfo, setPdfInfo] = useState<PDFInfo | null>(null)
  const [isCheckingPdf, setIsCheckingPdf] = useState(false)
  const { isAuthenticated } = useAuthStore()
  const queryClient = useQueryClient()

  // Check if paper is already saved
  useEffect(() => {
    const checkSaved = async () => {
      if (!isAuthenticated) return
      try {
        const response = await fetch(`/api/v1/library/papers/check/${paper.pmid}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
          }
        })
        if (response.ok) {
          const data = await response.json()
          setIsSaved(data.is_saved)
        }
      } catch (error) {
        logError('Check saved status', error)
      }
    }
    checkSaved()
  }, [paper.pmid, isAuthenticated])

  // Handle save paper
  const handleSavePaper = async () => {
    if (!isAuthenticated) {
      alert('로그인이 필요합니다')
      return
    }

    if (isSaved) {
      // Already saved - could implement unsave here
      return
    }

    setIsSaving(true)
    try {
      await libraryApi.savePaper(paper.pmid)
      setIsSaved(true)
      queryClient.invalidateQueries({ queryKey: ['savedPapers'] })
    } catch (error) {
      logError('Save paper', error)
      alert(getErrorMessage(error))
    } finally {
      setIsSaving(false)
    }
  }

  // Check PDF availability and download
  const handlePdfClick = async () => {
    // First check if PDF is available
    if (!pdfInfo) {
      setIsCheckingPdf(true)
      try {
        const info = await searchApi.getPdfInfo(paper.pmid)
        setPdfInfo(info)

        // If PDF is available, download it
        if (info.hasPdf && info.pdfUrl) {
          // Open PDF URL in new tab (PMC provides direct PDF links)
          window.open(info.pdfUrl, '_blank')
        }
      } catch (error) {
        logError('Check PDF availability', error)
        setPdfInfo({ pmid: paper.pmid, hasPdf: false, isOpenAccess: false })
      } finally {
        setIsCheckingPdf(false)
      }
    } else if (pdfInfo.hasPdf && pdfInfo.pdfUrl) {
      // Already checked, open PDF URL
      window.open(pdfInfo.pdfUrl, '_blank')
    }
  }

  // Fetch Korean summary from API when button is clicked
  const handleKoreanSummaryClick = async () => {
    // If already have summary, just toggle display
    if (koreanSummary) {
      setShowKorean(!showKorean)
      return
    }

    // Fetch Korean summary
    setIsLoadingSummary(true)
    setSummaryError(false)
    setShowKorean(true)

    try {
      const response = await searchApi.summarize(paper.abstract, 'ko')
      setKoreanSummary(response.summary)
    } catch (error) {
      logError('Fetch Korean summary', error)
      setSummaryError(true)
      setKoreanSummary(`[요약 실패] ${paper.abstract.substring(0, 200)}...`)
    } finally {
      setIsLoadingSummary(false)
    }
  }

  return (
    <div className="glossy-panel p-6 hover:scale-[1.01] transition-all duration-300">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <Link
            to={`/paper/${paper.pmid}`}
            className="text-lg font-semibold liquid-text hover:text-yellow-200 transition-colors"
          >
            {paper.title}
          </Link>
          <div className="flex items-center gap-3 mt-2 text-sm liquid-text-muted">
            <span>PMID: {paper.pmid}</span>
            <span>|</span>
            <span>{paper.journal}</span>
            {paper.publicationDate && (
              <>
                <span>|</span>
                <span>{paper.publicationDate}</span>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1 px-3 py-1 bg-green-500/30 text-green-200 rounded-full text-sm font-medium border border-green-400/30">
          {relevancePercent}%
        </div>
      </div>

      {/* Abstract with Korean Summary Toggle */}
      <div className="mt-4">
        {/* Toggle Button */}
        <div className="flex items-center gap-2 mb-2">
          <button
            onClick={handleKoreanSummaryClick}
            disabled={isLoadingSummary}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium transition-all ${
              showKorean
                ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
                : 'bg-white/10 text-white/70 border border-white/20 hover:bg-white/20'
            }`}
          >
            <Languages size={14} />
            {isLoadingSummary ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                <span>번역 중...</span>
              </>
            ) : showKorean ? (
              <span>영문 보기</span>
            ) : (
              <span>한글 요약</span>
            )}
          </button>
          {showKorean && koreanSummary && !isLoadingSummary && (
            <span className="text-xs text-cyan-300/70">AI 번역</span>
          )}
        </div>

        {/* Content */}
        {isLoadingSummary ? (
          <div className="bg-white/5 p-3 rounded-lg border border-white/10 animate-pulse">
            <div className="h-4 bg-white/10 rounded w-full mb-2"></div>
            <div className="h-4 bg-white/10 rounded w-5/6 mb-2"></div>
            <div className="h-4 bg-white/10 rounded w-3/4"></div>
          </div>
        ) : showKorean && koreanSummary ? (
          <p className={`liquid-text-muted line-clamp-4 bg-cyan-500/10 p-3 rounded-lg border ${summaryError ? 'border-red-400/30' : 'border-cyan-400/20'}`}>
            {koreanSummary}
          </p>
        ) : (
          <p className="liquid-text-muted line-clamp-3">
            {paper.abstract}
          </p>
        )}
      </div>

      {paper.authors.length > 0 && (
        <div className="mt-3 text-sm text-white/60">
          {paper.authors.slice(0, 3).join(', ')}
          {paper.authors.length > 3 && ` 외 ${paper.authors.length - 3}명`}
        </div>
      )}

      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-white/10">
        <Link
          to={`/paper/${paper.pmid}`}
          className="flex items-center gap-1 text-sm text-cyan-300 hover:text-cyan-200 transition-colors"
        >
          <ExternalLink size={16} />
          상세보기
        </Link>
        <Link
          to={`/chat?pmid=${paper.pmid}`}
          className="flex items-center gap-1 text-sm text-pink-300 hover:text-pink-200 transition-colors"
        >
          <MessageSquare size={16} />
          AI 질문
        </Link>
        <button
          onClick={handleSavePaper}
          disabled={isSaving || isSaved}
          className={`flex items-center gap-1 text-sm transition-colors ${
            isSaved
              ? 'text-yellow-300'
              : 'text-white/60 hover:text-yellow-300'
          }`}
        >
          {isSaving ? (
            <Loader2 size={16} className="animate-spin" />
          ) : isSaved ? (
            <BookmarkCheck size={16} />
          ) : (
            <Bookmark size={16} />
          )}
          {isSaved ? '저장됨' : '저장'}
        </button>
        <button
          onClick={handlePdfClick}
          disabled={isCheckingPdf || (pdfInfo !== null && !pdfInfo.hasPdf)}
          className={`flex items-center gap-1 text-sm transition-colors ${
            pdfInfo?.hasPdf
              ? 'text-green-300 hover:text-green-200'
              : pdfInfo !== null
                ? 'text-white/30 cursor-not-allowed'
                : 'text-white/60 hover:text-green-300'
          }`}
          title={
            pdfInfo?.hasPdf
              ? 'PDF 다운로드 (Open Access)'
              : pdfInfo !== null
                ? 'PDF 미제공'
                : 'PDF 확인'
          }
        >
          {isCheckingPdf ? (
            <Loader2 size={16} className="animate-spin" />
          ) : pdfInfo?.hasPdf ? (
            <FileDown size={16} />
          ) : pdfInfo !== null ? (
            <FileX size={16} />
          ) : (
            <FileDown size={16} />
          )}
          {isCheckingPdf
            ? '확인 중...'
            : pdfInfo?.hasPdf
              ? 'PDF'
              : pdfInfo !== null
                ? '미제공'
                : 'PDF'}
        </button>
      </div>
    </div>
  )
}
