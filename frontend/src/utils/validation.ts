/**
 * Input validation utilities for security
 */

/**
 * Sanitize user input by removing dangerous characters and limiting length
 */
export const sanitizeInput = (input: string, maxLength: number = 500): string => {
  const trimmed = input.trim().slice(0, maxLength)

  // Remove script tags and other dangerous patterns
  const sanitized = trimmed
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<[^>]*>/g, '') // Remove all HTML tags

  return sanitized
}

/**
 * Validate email format
 */
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email) && email.length <= 255
}

/**
 * Validate search query
 */
export const validateSearchQuery = (query: string): {
  valid: boolean
  error?: string
  sanitized?: string
} => {
  if (!query || query.trim().length === 0) {
    return { valid: false, error: '검색어를 입력하세요' }
  }

  const sanitized = sanitizeInput(query, 500)

  if (sanitized.length < 2) {
    return { valid: false, error: '검색어는 최소 2자 이상이어야 합니다' }
  }

  if (sanitized.length > 500) {
    return { valid: false, error: '검색어는 500자를 초과할 수 없습니다' }
  }

  return { valid: true, sanitized }
}

/**
 * Validate chat message
 */
export const validateChatMessage = (message: string): {
  valid: boolean
  error?: string
  sanitized?: string
} => {
  if (!message || message.trim().length === 0) {
    return { valid: false, error: '메시지를 입력하세요' }
  }

  const sanitized = sanitizeInput(message, 2000)

  if (sanitized.length < 1) {
    return { valid: false, error: '메시지를 입력하세요' }
  }

  if (sanitized.length > 2000) {
    return { valid: false, error: '메시지는 2000자를 초과할 수 없습니다' }
  }

  return { valid: true, sanitized }
}

/**
 * Validate year parameter from URL
 */
export const validateYearParam = (year: string | null): number | undefined => {
  if (!year) return undefined

  const parsed = parseInt(year, 10)

  if (isNaN(parsed)) return undefined

  const currentYear = new Date().getFullYear()
  if (parsed < 1900 || parsed > currentYear) return undefined

  return parsed
}

/**
 * Validate string array parameter from URL
 */
export const validateStringArrayParam = (
  param: string | null,
  maxItems: number = 10,
  maxLength: number = 100
): string[] | undefined => {
  if (!param) return undefined

  const items = param
    .split(',')
    .map((item) => sanitizeInput(item.trim(), maxLength))
    .filter(Boolean)
    .slice(0, maxItems)

  return items.length > 0 ? items : undefined
}

/**
 * Validate number is within range
 */
export const validateNumberRange = (
  value: number | undefined,
  min: number,
  max: number
): boolean => {
  if (value === undefined) return true
  return value >= min && value <= max
}
