/**
 * External Scientific Paper APIs
 * - arXiv: Physics, Math, CS, Engineering preprints
 * - CrossRef: DOI metadata for all disciplines
 * - Semantic Scholar: AI-powered research (rate limited)
 */

export interface ExternalPaper {
  id: string
  title: string
  authors: string[]
  abstract?: string
  year?: number
  url: string
  source: 'arxiv' | 'crossref' | 'semanticscholar'
  doi?: string
  categories?: string[]
}

export interface ExternalSearchResponse {
  papers: ExternalPaper[]
  total: number
  source: string
}

// arXiv API - Free, no authentication required
// Returns XML, needs parsing
export async function searchArxiv(query: string, maxResults: number = 10): Promise<ExternalSearchResponse> {
  try {
    const encodedQuery = encodeURIComponent(query)
    const url = `https://export.arxiv.org/api/query?search_query=all:${encodedQuery}&start=0&max_results=${maxResults}&sortBy=relevance&sortOrder=descending`

    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`arXiv API error: ${response.status}`)
    }

    const xmlText = await response.text()
    const parser = new DOMParser()
    const xmlDoc = parser.parseFromString(xmlText, 'text/xml')

    const entries = xmlDoc.querySelectorAll('entry')
    const totalResults = xmlDoc.querySelector('opensearch\\:totalResults, totalResults')?.textContent || '0'

    const papers: ExternalPaper[] = []

    entries.forEach((entry) => {
      const id = entry.querySelector('id')?.textContent || ''
      const arxivId = id.split('/abs/').pop() || id

      const authors: string[] = []
      entry.querySelectorAll('author name').forEach((author) => {
        if (author.textContent) authors.push(author.textContent)
      })

      const categories: string[] = []
      entry.querySelectorAll('category').forEach((cat) => {
        const term = cat.getAttribute('term')
        if (term) categories.push(term)
      })

      const published = entry.querySelector('published')?.textContent
      const year = published ? new Date(published).getFullYear() : undefined

      papers.push({
        id: arxivId,
        title: entry.querySelector('title')?.textContent?.replace(/\s+/g, ' ').trim() || '',
        authors,
        abstract: entry.querySelector('summary')?.textContent?.trim(),
        year,
        url: entry.querySelector('link[rel="alternate"]')?.getAttribute('href') || `https://arxiv.org/abs/${arxivId}`,
        source: 'arxiv',
        categories,
      })
    })

    return {
      papers,
      total: parseInt(totalResults),
      source: 'arXiv',
    }
  } catch (error) {
    console.error('arXiv search error:', error)
    return { papers: [], total: 0, source: 'arXiv' }
  }
}

// CrossRef API - Free, no authentication required
// Returns JSON, academic metadata
export async function searchCrossRef(query: string, maxResults: number = 10): Promise<ExternalSearchResponse> {
  try {
    const encodedQuery = encodeURIComponent(query)
    const url = `https://api.crossref.org/works?query=${encodedQuery}&rows=${maxResults}&select=DOI,title,author,abstract,published-print,URL,subject`

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Bio-RAG-App/1.0 (https://bio-rag.app; mailto:contact@bio-rag.app)',
      },
    })

    if (!response.ok) {
      throw new Error(`CrossRef API error: ${response.status}`)
    }

    const data = await response.json()
    const items = data.message?.items || []
    const total = data.message?.['total-results'] || 0

    const papers: ExternalPaper[] = items.map((item: {
      DOI?: string
      title?: string[]
      author?: { given?: string; family?: string }[]
      abstract?: string
      'published-print'?: { 'date-parts'?: number[][] }
      URL?: string
      subject?: string[]
    }) => {
      const authors = (item.author || []).map((a) =>
        [a.given, a.family].filter(Boolean).join(' ')
      )

      const dateParts = item['published-print']?.['date-parts']?.[0]
      const year = dateParts?.[0]

      // Clean abstract (remove HTML tags)
      let abstract = item.abstract || ''
      abstract = abstract.replace(/<[^>]*>/g, '').trim()

      return {
        id: item.DOI || '',
        title: item.title?.[0] || '',
        authors,
        abstract: abstract || undefined,
        year,
        url: item.URL || `https://doi.org/${item.DOI}`,
        source: 'crossref' as const,
        doi: item.DOI,
        categories: item.subject,
      }
    })

    return {
      papers,
      total,
      source: 'CrossRef',
    }
  } catch (error) {
    console.error('CrossRef search error:', error)
    return { papers: [], total: 0, source: 'CrossRef' }
  }
}

// Semantic Scholar API - Free but rate limited
// Returns JSON, AI-powered search
export async function searchSemanticScholar(query: string, maxResults: number = 10): Promise<ExternalSearchResponse> {
  try {
    const encodedQuery = encodeURIComponent(query)
    const fields = 'title,authors,year,abstract,url,externalIds'
    const url = `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodedQuery}&limit=${maxResults}&fields=${fields}`

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Bio-RAG-App/1.0',
      },
    })

    if (response.status === 429) {
      console.warn('Semantic Scholar rate limited')
      return { papers: [], total: 0, source: 'Semantic Scholar (Rate Limited)' }
    }

    if (!response.ok) {
      throw new Error(`Semantic Scholar API error: ${response.status}`)
    }

    const data = await response.json()
    const items = data.data || []
    const total = data.total || 0

    const papers: ExternalPaper[] = items.map((item: {
      paperId?: string
      title?: string
      authors?: { name?: string }[]
      year?: number
      abstract?: string
      url?: string
      externalIds?: { DOI?: string; ArXiv?: string }
    }) => ({
      id: item.paperId || '',
      title: item.title || '',
      authors: (item.authors || []).map((a) => a.name || ''),
      abstract: item.abstract,
      year: item.year,
      url: item.url || `https://www.semanticscholar.org/paper/${item.paperId}`,
      source: 'semanticscholar' as const,
      doi: item.externalIds?.DOI,
    }))

    return {
      papers,
      total,
      source: 'Semantic Scholar',
    }
  } catch (error) {
    console.error('Semantic Scholar search error:', error)
    return { papers: [], total: 0, source: 'Semantic Scholar' }
  }
}

// Combined search across all sources
export async function searchAllSources(
  query: string,
  sources: ('arxiv' | 'crossref' | 'semanticscholar')[] = ['arxiv', 'crossref'],
  maxResultsPerSource: number = 5
): Promise<{ results: ExternalSearchResponse[]; totalPapers: number }> {
  const searchPromises: Promise<ExternalSearchResponse>[] = []

  if (sources.includes('arxiv')) {
    searchPromises.push(searchArxiv(query, maxResultsPerSource))
  }
  if (sources.includes('crossref')) {
    searchPromises.push(searchCrossRef(query, maxResultsPerSource))
  }
  if (sources.includes('semanticscholar')) {
    searchPromises.push(searchSemanticScholar(query, maxResultsPerSource))
  }

  const results = await Promise.all(searchPromises)
  const totalPapers = results.reduce((sum, r) => sum + r.papers.length, 0)

  return { results, totalPapers }
}

export const externalApis = {
  searchArxiv,
  searchCrossRef,
  searchSemanticScholar,
  searchAllSources,
}

export default externalApis
