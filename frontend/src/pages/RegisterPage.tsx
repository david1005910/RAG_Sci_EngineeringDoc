import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { authApi } from '@/services/api'
import { useAuthStore } from '@/store/authStore'

const researchFields = [
  '암 연구',
  '면역학',
  '유전학',
  '신경과학',
  '약리학',
  '세포생물학',
  '분자생물학',
  '생화학',
  '기타',
]

export default function RegisterPage() {
  const navigate = useNavigate()
  const { setToken, setUser } = useAuthStore()
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    passwordConfirm: '',
    name: '',
    researchField: '',
  })
  const [error, setError] = useState('')

  const registerMutation = useMutation({
    mutationFn: () =>
      authApi.register(
        formData.email,
        formData.password,
        formData.name,
        formData.researchField
      ),
    onSuccess: async (data) => {
      setToken(data.accessToken)
      try {
        const user = await authApi.getMe()
        setUser(user)
      } catch {
        // User fetch failed
      }
      navigate('/')
    },
    onError: () => {
      setError('회원가입 중 오류가 발생했습니다. 다시 시도해주세요.')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (formData.password !== formData.passwordConfirm) {
      setError('비밀번호가 일치하지 않습니다.')
      return
    }

    if (formData.password.length < 8) {
      setError('비밀번호는 8자 이상이어야 합니다.')
      return
    }

    registerMutation.mutate()
  }

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }))
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
          <h1 className="mt-6 text-2xl font-bold liquid-text">회원가입</h1>
          <p className="mt-2 liquid-text-muted">새 계정을 만드세요</p>
        </div>

        <div className="glossy-panel p-8">
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="p-3 bg-red-500/20 border border-red-400/30 text-red-200 rounded-xl text-sm">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium liquid-text mb-2">
                이름
              </label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
                className="glossy-input w-full px-4 py-3"
                placeholder="홍길동"
              />
            </div>

            <div>
              <label className="block text-sm font-medium liquid-text mb-2">
                이메일
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
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
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                className="glossy-input w-full px-4 py-3"
                placeholder="8자 이상"
              />
            </div>

            <div>
              <label className="block text-sm font-medium liquid-text mb-2">
                비밀번호 확인
              </label>
              <input
                type="password"
                name="passwordConfirm"
                value={formData.passwordConfirm}
                onChange={handleChange}
                required
                className="glossy-input w-full px-4 py-3"
                placeholder="••••••••"
              />
            </div>

            <div>
              <label className="block text-sm font-medium liquid-text mb-2">
                연구 분야 (선택)
              </label>
              <select
                name="researchField"
                value={formData.researchField}
                onChange={handleChange}
                className="glossy-input w-full px-4 py-3"
              >
                <option value="">선택하세요</option>
                {researchFields.map((field) => (
                  <option key={field} value={field}>
                    {field}
                  </option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              disabled={registerMutation.isPending}
              className="glossy-btn-primary w-full py-3 font-medium disabled:opacity-50"
            >
              {registerMutation.isPending ? '가입 중...' : '회원가입'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <span className="liquid-text-muted">이미 계정이 있으신가요? </span>
            <Link to="/login" className="text-cyan-300 hover:text-cyan-200 transition-colors">
              로그인
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
