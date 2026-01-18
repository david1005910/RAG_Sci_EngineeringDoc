import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { authApi } from '@/services/api'
import { useAuthStore } from '@/store/authStore'

export default function LoginPage() {
  const navigate = useNavigate()
  const { setToken, setUser } = useAuthStore()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const loginMutation = useMutation({
    mutationFn: () => authApi.login(email, password),
    onSuccess: async (data) => {
      setToken(data.accessToken)
      try {
        const user = await authApi.getMe()
        setUser(user)
      } catch {
        // User fetch failed, but login succeeded
      }
      navigate('/')
    },
    onError: () => {
      setError('이메일 또는 비밀번호가 올바르지 않습니다.')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    loginMutation.mutate()
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-xl">B</span>
            </div>
            <span className="text-2xl font-bold liquid-text">Bio-RAG</span>
          </Link>
          <h1 className="mt-6 text-2xl font-bold liquid-text">로그인</h1>
          <p className="mt-2 liquid-text-muted">계정에 로그인하세요</p>
        </div>

        <div className="glossy-panel p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="p-3 bg-red-500/20 border border-red-400/30 text-red-200 rounded-xl text-sm">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium liquid-text mb-2">
                이메일
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="glossy-input w-full px-4 py-3"
                placeholder="your@email.com"
              />
            </div>

            <div>
              <label className="block text-sm font-medium liquid-text mb-2">
                비밀번호
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="glossy-input w-full px-4 py-3"
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={loginMutation.isPending}
              className="glossy-btn-primary w-full py-3 font-medium disabled:opacity-50"
            >
              {loginMutation.isPending ? '로그인 중...' : '로그인'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <span className="liquid-text-muted">계정이 없으신가요? </span>
            <Link to="/register" className="text-cyan-300 hover:text-cyan-200 transition-colors">
              회원가입
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
