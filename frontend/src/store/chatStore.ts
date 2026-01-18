import { create } from 'zustand'
import type { ChatMessage, ChatSession } from '@/types'

interface ChatState {
  currentSessionId: string | null
  messages: ChatMessage[]
  sessions: ChatSession[]
  isLoading: boolean

  setCurrentSession: (sessionId: string | null) => void
  addMessage: (message: ChatMessage) => void
  setMessages: (messages: ChatMessage[]) => void
  setSessions: (sessions: ChatSession[]) => void
  setLoading: (loading: boolean) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  currentSessionId: null,
  messages: [],
  sessions: [],
  isLoading: false,

  setCurrentSession: (sessionId) =>
    set({ currentSessionId: sessionId }),

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  setMessages: (messages) =>
    set({ messages }),

  setSessions: (sessions) =>
    set({ sessions }),

  setLoading: (isLoading) =>
    set({ isLoading }),

  clearMessages: () =>
    set({ messages: [], currentSessionId: null }),
}))
