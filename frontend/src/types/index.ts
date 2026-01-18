// Paper types
export interface Paper {
  pmid: string
  title: string
  abstract: string
  authors: string[]
  journal: string
  publicationDate?: string
  doi?: string
  keywords: string[]
  meshTerms?: string[]
}

export interface PaperSearchResult {
  pmid: string
  title: string
  abstract: string
  relevanceScore: number
  authors: string[]
  journal: string
  publicationDate?: string
  keywords: string[]
}

export interface SearchResponse {
  total: number
  tookMs: number
  results: PaperSearchResult[]
}

// Chat types
export interface ChatSource {
  pmid: string
  title: string
  relevance: number
  excerpt: string
  sourceType?: string
  denseScore?: number
  sparseScore?: number
}

export interface SimilarPaper {
  pmid: string
  title: string
  similarity_score?: number
  common_keywords?: string[]
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
  createdAt: string
}

export interface ChatSession {
  id: string
  title: string
  createdAt: string
  messageCount: number
}

export interface ChatQueryResponse {
  answer: string
  sources: ChatSource[]
  confidence: number
  processingTimeMs: number
  sessionId: string
}

// User types
export interface User {
  id: string
  email: string
  name: string
  researchField?: string
}

export interface AuthTokens {
  accessToken: string
  tokenType: string
  expiresIn: number
}

// Saved paper types
export interface SavedPaper {
  id: string
  pmid: string
  title: string
  abstract: string
  journal?: string
  tags: string[]
  notes?: string
  savedAt: string
}

// Trend types
export interface KeywordTrend {
  date: string
  keyword: string
  count: number
}

export interface HotTopic {
  keyword: string
  count: number
  growthRate: number
}

// PDF types
export interface PDFInfo {
  pmid: string
  pmcid?: string
  hasPdf: boolean
  pdfUrl?: string
  isOpenAccess: boolean
}
