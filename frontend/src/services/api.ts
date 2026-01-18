import axios from 'axios'
import type {
  SearchResponse,
  Paper,
  ChatQueryResponse,
  ChatSession,
  ChatMessage,
  ChatSource,
  AuthTokens,
  User,
  SavedPaper,
  HotTopic,
  KeywordTrend,
  PDFInfo,
  SimilarPaper,
} from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// CSRF token cookie name (must match backend)
const CSRF_COOKIE_NAME = 'csrf_token'

/**
 * Get CSRF token from cookie
 */
function getCsrfToken(): string | null {
  const cookies = document.cookie.split(';')
  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split('=')
    if (name === CSRF_COOKIE_NAME) {
      return decodeURIComponent(value)
    }
  }
  return null
}

// Create axios instance
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Send cookies with requests
})

// Request interceptor for auth token and CSRF
api.interceptors.request.use((config) => {
  // Add auth token
  const token = localStorage.getItem('accessToken')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  // Add CSRF token for state-changing requests
  const method = config.method?.toUpperCase()
  if (method && ['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
    const csrfToken = getCsrfToken()
    if (csrfToken) {
      config.headers['X-CSRF-Token'] = csrfToken
    }
  }

  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('accessToken')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

/**
 * Initialize CSRF token by making a request to get the cookie set
 */
export async function initCsrfToken(): Promise<void> {
  try {
    await api.get('/csrf-token')
  } catch (error) {
    console.warn('Failed to initialize CSRF token:', error)
  }
}

// ============== Auth API ==============

export const authApi = {
  login: async (email: string, password: string): Promise<AuthTokens> => {
    const response = await api.post('/auth/login', { email, password })
    return {
      accessToken: response.data.access_token,
      tokenType: response.data.token_type,
      expiresIn: response.data.expires_in,
    }
  },

  register: async (
    email: string,
    password: string,
    name: string,
    researchField?: string
  ): Promise<AuthTokens> => {
    const response = await api.post('/auth/register', {
      email,
      password,
      name,
      research_field: researchField,
    })
    return {
      accessToken: response.data.access_token,
      tokenType: response.data.token_type,
      expiresIn: response.data.expires_in,
    }
  },

  logout: async (): Promise<void> => {
    await api.post('/auth/logout')
    localStorage.removeItem('accessToken')
  },

  getMe: async (): Promise<User> => {
    const response = await api.get('/auth/me')
    return {
      id: response.data.id,
      email: response.data.email,
      name: response.data.name,
      researchField: response.data.research_field,
    }
  },
}

// ============== Search API ==============

export const searchApi = {
  search: async (
    query: string,
    limit: number = 10,
    filters?: Record<string, unknown>,
    source: 'pubmed' | 'mock' = 'pubmed'
  ): Promise<SearchResponse> => {
    const response = await api.post('/search', { query, limit, filters, source })
    return {
      total: response.data.total,
      tookMs: response.data.took_ms,
      results: response.data.results.map((r: Record<string, unknown>) => ({
        pmid: r.pmid,
        title: r.title,
        abstract: r.abstract,
        relevanceScore: r.relevance_score,
        authors: r.authors,
        journal: r.journal,
        publicationDate: r.publication_date,
        keywords: r.keywords,
      })),
    }
  },

  getPaper: async (pmid: string): Promise<Paper> => {
    const response = await api.get(`/papers/${pmid}`)
    const r = response.data
    return {
      pmid: r.pmid,
      title: r.title,
      abstract: r.abstract,
      authors: r.authors || [],
      journal: r.journal,
      publicationDate: r.publication_date,
      doi: r.doi,
      keywords: r.keywords || [],
      meshTerms: r.mesh_terms || [],
    }
  },

  getSimilarPapers: async (pmid: string, limit: number = 5): Promise<SimilarPaper[]> => {
    const response = await api.get(`/papers/${pmid}/similar`, { params: { limit } })
    // API returns array directly
    return response.data || []
  },

  getPdfInfo: async (pmid: string): Promise<PDFInfo> => {
    const response = await api.get(`/papers/${pmid}/pdf-info`)
    return {
      pmid: response.data.pmid,
      pmcid: response.data.pmcid,
      hasPdf: response.data.has_pdf,
      pdfUrl: response.data.pdf_url,
      isOpenAccess: response.data.is_open_access,
    }
  },

  getPdfInfoBatch: async (pmids: string[]): Promise<PDFInfo[]> => {
    const response = await api.post('/papers/pdf-info-batch', { pmids })
    return response.data.papers.map((p: Record<string, unknown>) => ({
      pmid: p.pmid,
      pmcid: p.pmcid,
      hasPdf: p.has_pdf,
      pdfUrl: p.pdf_url,
      isOpenAccess: p.is_open_access,
    }))
  },

  downloadPdf: async (pmid: string): Promise<Blob> => {
    const response = await api.get(`/papers/${pmid}/pdf`, {
      responseType: 'blob',
    })
    return response.data
  },

  summarize: async (text: string, language: string = 'ko'): Promise<{ summary: string }> => {
    const response = await api.post('/summarize', { text, language })
    return {
      summary: response.data.summary,
    }
  },

  translate: async (text: string, sourceLang: string = 'ko', targetLang: string = 'en'): Promise<{ original: string; translated: string }> => {
    const response = await api.post('/translate', {
      text,
      source_lang: sourceLang,
      target_lang: targetLang,
    })
    return {
      original: response.data.original,
      translated: response.data.translated,
    }
  },
}

// ============== Chat API ==============

export interface ChatQueryOptions {
  question: string
  sessionId?: string
  contextPmids?: string[]
  useVectordb?: boolean
  searchMode?: 'hybrid' | 'dense' | 'sparse'
  denseWeight?: number
}

export interface ChatQueryResponseExtended extends ChatQueryResponse {
  vectordbUsed: boolean
  searchMode?: string
}

// API response source (snake_case from backend)
interface ApiChatSource {
  pmid: string
  title: string
  relevance: number
  excerpt: string
  source_type?: string
  dense_score?: number
  sparse_score?: number
}

export const chatApi = {
  query: async (
    question: string,
    sessionId?: string,
    contextPmids?: string[],
    options?: {
      useVectordb?: boolean
      searchMode?: 'hybrid' | 'dense' | 'sparse'
      denseWeight?: number
    }
  ): Promise<ChatQueryResponseExtended> => {
    const response = await api.post('/chat/query', {
      question,
      session_id: sessionId,
      context_pmids: contextPmids,
      use_vectordb: options?.useVectordb ?? true,
      search_mode: options?.searchMode ?? 'hybrid',
      dense_weight: options?.denseWeight ?? 0.7,
    })
    return {
      answer: response.data.answer,
      sources: response.data.sources.map((s: ApiChatSource): ChatSource => ({
        ...s,
        sourceType: s.source_type,
        denseScore: s.dense_score,
        sparseScore: s.sparse_score,
      })),
      confidence: response.data.confidence,
      processingTimeMs: response.data.processing_time_ms,
      sessionId: response.data.session_id,
      vectordbUsed: response.data.vectordb_used,
      searchMode: response.data.search_mode,
    }
  },

  getSessions: async (): Promise<ChatSession[]> => {
    const response = await api.get('/chat/sessions')
    return response.data.sessions
  },

  getSessionMessages: async (sessionId: string): Promise<ChatMessage[]> => {
    const response = await api.get(`/chat/sessions/${sessionId}/messages`)
    return response.data.messages
  },

  deleteSession: async (sessionId: string): Promise<void> => {
    await api.delete(`/chat/sessions/${sessionId}`)
  },
}

// ============== Library API ==============

export const libraryApi = {
  getSavedPapers: async (tag?: string): Promise<SavedPaper[]> => {
    const response = await api.get('/library/papers', { params: { tag } })
    return response.data.papers
  },

  savePaper: async (
    pmid: string,
    tags?: string[],
    notes?: string
  ): Promise<SavedPaper> => {
    const response = await api.post('/library/papers', { pmid, tags, notes })
    return response.data
  },

  deleteSavedPaper: async (paperId: string): Promise<void> => {
    await api.delete(`/library/papers/${paperId}`)
  },

  getTags: async (): Promise<string[]> => {
    const response = await api.get('/library/tags')
    return response.data.tags
  },
}

// ============== Trends API ==============

export interface TrendAnalysis {
  query: string
  analysis: string
  keyTrends: string[]
  relatedTopics: string[]
  researchDirection: string
  summary: string
}

export const trendsApi = {
  getKeywordTrends: async (keywords: string[]): Promise<KeywordTrend[]> => {
    const params = new URLSearchParams()
    keywords.forEach(k => params.append('keywords', k))
    const response = await api.get(`/trends/keywords?${params.toString()}`)
    return response.data.data
  },

  getHotTopics: async (limit: number = 10): Promise<HotTopic[]> => {
    const response = await api.get('/trends/hot', { params: { limit } })
    return response.data.topics.map((t: { keyword: string; count: number; growth_rate: number }) => ({
      keyword: t.keyword,
      count: t.count,
      growthRate: t.growth_rate,
    }))
  },

  analyzeTrend: async (query: string, language: string = 'ko'): Promise<TrendAnalysis> => {
    const response = await api.post('/trends/analyze', { query, language })
    return {
      query: response.data.query,
      analysis: response.data.analysis,
      keyTrends: response.data.key_trends,
      relatedTopics: response.data.related_topics,
      researchDirection: response.data.research_direction,
      summary: response.data.summary,
    }
  },
}

// ============== VectorDB API ==============

export interface PaperForVectorDB {
  pmid: string
  title: string
  abstract: string
  authors: string[]
  journal: string
  publication_date?: string
  keywords: string[]
}

export interface SavePapersResponse {
  saved_count: number
  total_chunks: number
  processing_time_ms: number
  paper_ids: string[]
}

export interface VectorDBStats {
  collection_name: string
  vectors_count: number
  status: string
}

export interface VectorSearchResult {
  pmid: string
  title: string
  text: string
  score: number
  dense_score?: number
  sparse_score?: number
  section: string
}

export interface VectorSearchResponse {
  results: VectorSearchResult[]
  took_ms: number
  search_mode: string
}

export interface VectorDBPaper {
  id: string
  pmid: string
  title: string
  abstract: string
  journal?: string
  authors: string[]
  keywords: string[]
  indexed_at?: string
}

export interface VectorDBPapersResponse {
  papers: VectorDBPaper[]
  total: number
}

export const vectordbApi = {
  savePapers: async (papers: PaperForVectorDB[]): Promise<SavePapersResponse> => {
    const response = await api.post('/vectordb/papers/save', { papers })
    return response.data
  },

  getPapers: async (): Promise<VectorDBPapersResponse> => {
    const response = await api.get('/vectordb/papers')
    return response.data
  },

  getMetadata: async (): Promise<VectorDBPapersResponse> => {
    const response = await api.get('/vectordb/metadata')
    return response.data
  },

  getStats: async (): Promise<VectorDBStats> => {
    const response = await api.get('/vectordb/stats')
    return response.data
  },

  search: async (
    query: string,
    topK: number = 5,
    searchMode: 'hybrid' | 'dense' | 'sparse' = 'hybrid',
    denseWeight: number = 0.7
  ): Promise<VectorSearchResponse> => {
    const response = await api.post('/vectordb/search', {
      query,
      top_k: topK,
      search_mode: searchMode,
      dense_weight: denseWeight
    })
    return response.data
  },

  clear: async (): Promise<void> => {
    await api.delete('/vectordb/clear')
  },
}
