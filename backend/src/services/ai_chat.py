"""AI Chat Service - OpenAI/Anthropic Integration with RAG"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import aiohttp

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChatSource:
    """Source document used for RAG"""
    pmid: str
    title: str
    abstract: str
    relevance: float


@dataclass
class ChatResponse:
    """AI chat response"""
    answer: str
    sources_used: List[str]  # PMIDs
    confidence: float


class AIService:
    """AI Service for chat with RAG support"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        provider: str = "openai"  # "openai" or "anthropic"
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model
        self.provider = provider
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_system_prompt(self) -> str:
        """Build system prompt for biomedical RAG with enhanced Chain of Thought"""
        return """You are Bio-RAG, a friendly and knowledgeable AI research assistant specialized in biomedical literature analysis.

🎯 **Your Mission:**
You help researchers, students, and healthcare professionals understand complex biomedical research by providing clear, thorough, and well-reasoned answers based on scientific literature.

📚 **Core Principles:**
1. **Accuracy First**: Always base your answers on the provided research papers
2. **Clear Citations**: Reference papers using PMID (e.g., "PMID:12345에 따르면..." or "According to PMID:12345...")
3. **Honest Limitations**: If information is incomplete, acknowledge it openly
4. **Accessible Language**: Explain complex concepts in an understandable way while maintaining scientific accuracy

🌐 **Language:**
- **IMPORTANT**: Respond in the SAME LANGUAGE as the user's question
- Korean question (한국어) → Korean response
- English question → English response
- Use appropriate scientific terminology with explanations when needed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 **RESPONSE FORMAT (Chain of Thought - 반드시 따라주세요):**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

차근차근 단계별로 사고하며 최선의 답변을 작성해주세요:

---

## 🔍 1. 질문 이해 (Understanding the Question)

사용자의 질문을 깊이 이해하고 명확하게 재진술합니다.
- 핵심 키워드와 개념 파악
- 질문의 범위와 맥락 이해
- 사용자가 정말 알고 싶어하는 것이 무엇인지 파악

---

## 💭 2. 사고 과정 (Thinking Process)

질문에 답하기 위한 접근 방식을 체계적으로 생각합니다.

**🧠 분석 관점:**
- 이 질문에 답하려면 어떤 정보가 필요한가?
- 제공된 논문들에서 어떤 관련 정보를 찾을 수 있는가?
- 여러 논문의 정보를 어떻게 종합할 것인가?

**📊 고려사항:**
- 연구 방법론의 신뢰성
- 결과의 일관성 또는 상충점
- 임상적/실제적 의미

---

## 🔬 3. 분석 및 관찰 (Analysis & Observations)

제공된 논문들을 하나씩 분석하고 관련 정보를 추출합니다.
(이 과정은 논문 수에 따라 여러 번 반복됩니다)

### 📄 논문 분석 1
- **출처**: [PMID 인용]
- **핵심 발견**: [주요 연구 결과]
- **방법론**: [연구 방법 간략 설명]
- **의의**: [이 논문이 질문에 어떻게 기여하는지]

### 📄 논문 분석 2
- **출처**: [PMID 인용]
- **핵심 발견**: [주요 연구 결과]
- **연관성**: [첫 번째 논문과의 연결점 또는 차이점]

(필요시 추가 논문 분석 계속...)

### 🔗 종합 관찰
- 논문들 간의 공통점과 차이점
- 전체적인 연구 동향
- 남아있는 불확실성이나 연구 격차

---

## ✨ 4. 최종 답변 (Final Answer)

### 📌 핵심 요약
[질문에 대한 직접적이고 명확한 답변 - 2-3문장]

### 📖 상세 설명
[위의 분석을 바탕으로 한 포괄적인 설명]
- 주요 연구 결과들의 종합
- 실제적/임상적 의미
- PMID 인용을 통한 근거 제시

### ⚠️ 참고사항
[해당되는 경우]
- 연구의 한계점
- 추가 연구가 필요한 부분
- 개인적 상황에 따른 고려사항

### 📚 인용된 논문
- PMID:XXXXX - [논문 제목 요약]
- PMID:XXXXX - [논문 제목 요약]

---

💡 **Remember**:
- 친근하면서도 전문적인 톤을 유지하세요
- 복잡한 개념은 비유나 예시를 들어 설명하세요
- 불확실한 부분은 솔직하게 인정하세요
- 항상 근거 기반의 답변을 제공하세요"""

    def _build_context_prompt(self, question: str, sources: List[ChatSource]) -> str:
        """Build context-aware prompt with paper information"""
        context_parts = ["Here are relevant research papers to help answer the question:\n"]

        for i, source in enumerate(sources, 1):
            context_parts.append(f"""
Paper {i}:
- PMID: {source.pmid}
- Title: {source.title}
- Abstract: {source.abstract}
- Relevance Score: {source.relevance:.2f}
""")

        context_parts.append(f"\nUser Question: {question}")
        context_parts.append("\nPlease provide a comprehensive answer based on the above papers, citing PMIDs where appropriate.")

        return "".join(context_parts)

    async def chat_with_context(
        self,
        question: str,
        sources: List[ChatSource],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """
        Generate AI response with RAG context

        Args:
            question: User's question
            sources: List of relevant papers as context
            conversation_history: Optional previous messages

        Returns:
            ChatResponse with answer and metadata
        """
        if not self.api_key:
            logger.warning("No API key configured, using fallback response")
            return self._fallback_response(question, sources)

        try:
            if self.provider == "openai":
                return await self._openai_chat(question, sources, conversation_history)
            elif self.provider == "anthropic":
                return await self._anthropic_chat(question, sources, conversation_history)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return self._fallback_response(question, sources)

    async def _openai_chat(
        self,
        question: str,
        sources: List[ChatSource],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """Call OpenAI API"""
        session = await self._get_session()

        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Last 6 messages for context

        # Add current question with paper context
        user_message = self._build_context_prompt(question, sources)
        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 4000
        }

        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenAI API error: {response.status} - {error_text}")
                return self._fallback_response(question, sources)

            data = await response.json()
            answer = data["choices"][0]["message"]["content"]

            # Extract which PMIDs were actually used in the response
            sources_used = [s.pmid for s in sources if s.pmid in answer]

            # Calculate confidence based on sources used and response quality
            confidence = min(0.95, 0.5 + (len(sources_used) * 0.1))

            return ChatResponse(
                answer=answer,
                sources_used=sources_used,
                confidence=confidence
            )

    async def _anthropic_chat(
        self,
        question: str,
        sources: List[ChatSource],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """Call Anthropic Claude API"""
        session = await self._get_session()

        # Build messages for Claude
        messages = []

        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        user_message = self._build_context_prompt(question, sources)
        messages.append({"role": "user", "content": user_message})

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": "claude-3-haiku-20240307",  # Fast and cost-effective
            "max_tokens": 4000,
            "temperature": 0.6,
            "system": self._build_system_prompt(),
            "messages": messages
        }

        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Anthropic API error: {response.status} - {error_text}")
                return self._fallback_response(question, sources)

            data = await response.json()
            answer = data["content"][0]["text"]

            sources_used = [s.pmid for s in sources if s.pmid in answer]
            confidence = min(0.95, 0.5 + (len(sources_used) * 0.1))

            return ChatResponse(
                answer=answer,
                sources_used=sources_used,
                confidence=confidence
            )

    def _fallback_response(self, question: str, sources: List[ChatSource]) -> ChatResponse:
        """Generate fallback response when AI is unavailable"""
        if not sources:
            return ChatResponse(
                answer="I couldn't find relevant papers for your question. Please try rephrasing or use different keywords.",
                sources_used=[],
                confidence=0.1
            )

        # Build a basic response from the sources
        answer_parts = [
            f"Based on my search, I found {len(sources)} relevant papers for your question about '{question[:50]}...':\n\n"
        ]

        for i, source in enumerate(sources[:3], 1):
            answer_parts.append(
                f"**{i}. {source.title}** (PMID: {source.pmid})\n"
                f"{source.abstract[:300]}...\n\n"
            )

        answer_parts.append(
            "\n*Note: AI-powered analysis is currently unavailable. "
            "Please refer to the full papers for detailed information.*"
        )

        return ChatResponse(
            answer="".join(answer_parts),
            sources_used=[s.pmid for s in sources[:3]],
            confidence=0.3
        )


# Global service instance
_ai_service: Optional[AIService] = None


def get_ai_service() -> AIService:
    """Get or create AI service instance"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL if hasattr(settings, 'OPENAI_MODEL') else "gpt-4o-mini"
        )
    return _ai_service
