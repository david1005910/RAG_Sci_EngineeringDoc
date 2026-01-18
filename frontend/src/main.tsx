import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { initCsrfToken } from './services/api'
import './index.css'

// Initialize CSRF token on app load
initCsrfToken()

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 10, // 10 minutes (increased)
      gcTime: 1000 * 60 * 60, // 1 hour garbage collection time
      retry: 1,
      refetchOnMount: false, // Don't refetch when component mounts (preserves cache on navigation)
      refetchOnWindowFocus: false, // Don't refetch when window gains focus
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
