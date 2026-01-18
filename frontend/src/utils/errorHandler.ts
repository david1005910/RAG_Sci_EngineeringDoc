/**
 * Error handling utilities for security and UX
 */

import axios from 'axios'

/**
 * Handle API errors safely without exposing sensitive information
 */
export const handleApiError = (error: unknown, userMessage: string): string => {
  // Only log detailed errors in development
  if (import.meta.env.DEV) {
    console.error('API Error:', error)
  }

  // In production, could send to error tracking service
  // if (import.meta.env.PROD) {
  //   logErrorToService(error)
  // }

  return userMessage
}

/**
 * Get user-friendly error message based on error type
 */
export const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    switch (error.response?.status) {
      case 400:
        return '잘못된 요청입니다'
      case 401:
        return '인증이 필요합니다. 다시 로그인해주세요'
      case 403:
        return '접근 권한이 없습니다'
      case 404:
        return '요청한 리소스를 찾을 수 없습니다'
      case 429:
        return '너무 많은 요청이 발생했습니다. 잠시 후 다시 시도하세요'
      case 500:
        return '서버 오류가 발생했습니다. 잠시 후 다시 시도하세요'
      case 502:
      case 503:
      case 504:
        return '서버에 연결할 수 없습니다. 잠시 후 다시 시도하세요'
      default:
        return '알 수 없는 오류가 발생했습니다'
    }
  }

  if (error instanceof Error) {
    if (error.message.includes('Network Error')) {
      return '네트워크 연결을 확인하세요'
    }
    if (error.message.includes('timeout')) {
      return '요청 시간이 초과되었습니다. 다시 시도하세요'
    }
  }

  return '요청 처리 중 오류가 발생했습니다'
}

/**
 * Safe error logging that doesn't expose sensitive data
 */
export const logError = (context: string, error: unknown): void => {
  if (import.meta.env.DEV) {
    console.error(`[${context}]`, error)
  }
  // Production: send sanitized error to monitoring service
}
