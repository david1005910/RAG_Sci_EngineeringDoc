import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { PaperSearchResult } from '@/types'

interface SearchFilters {
  yearFrom?: number
  yearTo?: number
  journals?: string[]
  authors?: string[]
}

interface SearchState {
  // Last search query
  lastQuery: string
  // Last search results
  lastResults: PaperSearchResult[] | null
  // Last filters
  lastFilters: SearchFilters
  // Current page
  currentPage: number
  // Auto-saved query tracking
  autoSavedQuery: string | null

  // Actions
  setLastSearch: (query: string, results: PaperSearchResult[], filters: SearchFilters) => void
  setCurrentPage: (page: number) => void
  setAutoSavedQuery: (query: string | null) => void
  clearSearch: () => void
}

export const useSearchStore = create<SearchState>()(
  persist(
    (set) => ({
      lastQuery: '',
      lastResults: null,
      lastFilters: {},
      currentPage: 1,
      autoSavedQuery: null,

      setLastSearch: (query, results, filters) =>
        set({
          lastQuery: query,
          lastResults: results,
          lastFilters: filters,
          currentPage: 1,
        }),

      setCurrentPage: (page) => set({ currentPage: page }),

      setAutoSavedQuery: (query) => set({ autoSavedQuery: query }),

      clearSearch: () =>
        set({
          lastQuery: '',
          lastResults: null,
          lastFilters: {},
          currentPage: 1,
          autoSavedQuery: null,
        }),
    }),
    {
      name: 'bio-rag-search-storage',
      // Only persist essential data, not the full results (too large)
      partialize: (state) => ({
        lastQuery: state.lastQuery,
        lastFilters: state.lastFilters,
        currentPage: state.currentPage,
        autoSavedQuery: state.autoSavedQuery,
      }),
    }
  )
)
