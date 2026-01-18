import { Outlet, Link, useLocation } from 'react-router-dom'
import { Search, MessageSquare, Library, TrendingUp, Menu, X } from 'lucide-react'
import { useState } from 'react'
import { useAuthStore } from '@/store/authStore'
// DNAHelix removed for cleaner UI

const navItems = [
  { path: '/chat', label: 'AI 챗봇', icon: MessageSquare },
  { path: '/search', label: '논문 검색', icon: Search },
  { path: '/library', label: '내 라이브러리', icon: Library },
  { path: '/trends', label: '트렌드', icon: TrendingUp },
]

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const location = useLocation()
  const { isAuthenticated, user, logout } = useAuthStore()

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* SVG Gooey Filter */}
      <svg style={{position:'absolute',width:0,height:0}}>
        <filter id="goo">
          <feGaussianBlur in="SourceGraphic" stdDeviation="10" result="blur"/>
          <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 19 -9" result="goo"/>
        </filter>
      </svg>

      {/* Spline 3D DNA Background - positioned far left */}
      <div className="fixed inset-0 -z-20 overflow-hidden">
        <iframe
          src="https://my.spline.design/dnaparticles-zmvjvOLq5IukMI1j9JLf0gJg/"
          frameBorder="0"
          width="100%"
          height="100%"
          style={{
            position: 'absolute',
            top: 0,
            left: '-50%',
            width: '150%',
            height: '100%',
            pointerEvents: 'none'
          }}
          title="DNA Particles Background"
        />
        {/* Cover Spline watermark */}
        <div className="absolute bottom-0 right-0 w-80 h-24 bg-gradient-to-tl from-[#fef3e2] via-[#fef3e2] to-transparent" />
      </div>

      {/* Animated Blobs Background (overlay) */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="blob blob-pink w-96 h-96 top-10 left-10 animate-blob opacity-40" />
        <div className="blob blob-purple w-80 h-80 top-40 right-20 animate-blob-delay-2 opacity-30" />
        <div className="blob blob-blue w-72 h-72 bottom-20 left-1/4 animate-blob-delay-4 opacity-25" />
        <div className="blob blob-orange w-64 h-64 bottom-40 right-1/3 animate-blob opacity-20" />
        <div className="blob blob-cyan w-56 h-56 top-1/2 left-1/2 animate-blob-delay-2 opacity-25" />
      </div>

      {/* Header */}
      <header className="glossy-header mx-4 mt-4 sticky top-4 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white font-bold text-lg">B</span>
              </div>
              <span className="text-xl font-bold liquid-text">Bio-RAG</span>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-1">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-2xl transition-all duration-300 ${
                      isActive
                        ? 'glossy-btn-primary'
                        : 'liquid-text-muted hover:bg-white/10'
                    }`}
                  >
                    <Icon size={18} />
                    <span>{item.label}</span>
                  </Link>
                )
              })}
            </nav>

            {/* Auth buttons */}
            <div className="hidden md:flex items-center space-x-4">
              {isAuthenticated ? (
                <>
                  <span className="text-sm liquid-text-muted">
                    {user?.name || user?.email}
                  </span>
                  <button
                    onClick={logout}
                    className="text-sm liquid-text-muted hover:text-white transition-colors"
                  >
                    로그아웃
                  </button>
                </>
              ) : (
                <>
                  <Link
                    to="/login"
                    className="text-sm liquid-text-muted hover:text-white transition-colors"
                  >
                    로그인
                  </Link>
                  <Link
                    to="/register"
                    className="glossy-btn-primary px-4 py-2 text-sm font-medium"
                  >
                    회원가입
                  </Link>
                </>
              )}
            </div>

            {/* Mobile menu button */}
            <button
              className="md:hidden p-2 liquid-text"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-white/20">
            <nav className="px-4 py-2 space-y-1">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center space-x-2 px-4 py-3 rounded-2xl transition-all ${
                      isActive
                        ? 'glossy-btn-primary'
                        : 'liquid-text-muted hover:bg-white/10'
                    }`}
                  >
                    <Icon size={18} />
                    <span>{item.label}</span>
                  </Link>
                )
              })}
            </nav>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 relative z-10">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="glossy-panel-sm mx-4 mb-4 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm liquid-text-muted">
            Bio-RAG - AI-powered biomedical research platform
          </p>
        </div>
      </footer>
    </div>
  )
}
