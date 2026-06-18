import re
from enum import Enum
from langchain_core.language_models.chat_models import BaseChatModel

class QueryIntent(str, Enum):
    RAG = "rag"
    CHITCHAT = "chitchat"

class SemanticRouter:
    """
    Router phân loại câu hỏi người dùng thành 2 luồng: 
    - CHITCHAT: Giao tiếp thường nhật
    - RAG: Cần tra cứu kiến thức pháp luật
    """
    def __init__(self, llm: BaseChatModel):
        # Truyền LLM instance vào để có thể dùng chung connection với hệ thống
        self.llm = llm
        self.router_prompt_template = """Bạn là một hệ thống phân loại ý định người dùng.
Câu hỏi của người dùng: "{query}"

Nhiệm vụ:
- Nếu đây là câu hỏi giao tiếp thông thường (chào hỏi, cảm ơn, khen ngợi, hỏi thăm, tạm biệt) hoặc câu không chứa nội dung cần tra cứu -> Trả lời "CHITCHAT".
- Nếu đây là câu hỏi cần tra cứu kiến thức, luật pháp, giao thông, mức phạt, hoặc bất kỳ thông tin chuyên môn nào -> Trả lời "RAG".

Chỉ trả lời duy nhất một từ "CHITCHAT" hoặc "RAG", tuyệt đối không giải thích thêm."""

    _LEGAL_HINTS = (
        "phạt",
        "luật",
        "nghị định",
        "điều",
        "khoản",
        "điểm",
        "vi phạm",
        "giao thông",
        "xe",
        "mũ bảo hiểm",
        "đèn đỏ",
        "nồng độ cồn",
        "giấy phép",
        "bằng lái",
        "tốc độ",
        "làn đường",
        "quá tải",
        "chở",
    )
    _CHITCHAT_HINTS = (
        "xin chào",
        "chào",
        "hello",
        "hi",
        "alo",
        "cảm ơn",
        "cam on",
        "thanks",
        "thank you",
        "tạm biệt",
        "tam biet",
        "bye",
        "bạn khỏe không",
        "ban khoe khong",
        "bạn là ai",
        "ban la ai",
        "ok",
        "oke",
        "test",
    )

    def _rule_based_route(self, query: str) -> QueryIntent | None:
        """Nhận diện nhanh các câu giao tiếp ngắn để không cần gọi LLM router."""
        normalized = " ".join(str(query or "").lower().split())
        if not normalized:
            return QueryIntent.CHITCHAT

        if any(hint in normalized for hint in self._LEGAL_HINTS):
            return None

        cleaned = re.sub(r"[^\w\s]", " ", normalized)
        cleaned = " ".join(cleaned.split())
        if len(cleaned) <= 80 and any(hint in cleaned for hint in self._CHITCHAT_HINTS):
            return QueryIntent.CHITCHAT

        return None

    def route(self, query: str) -> QueryIntent:
        rule_intent = self._rule_based_route(query)
        if rule_intent is not None:
            return rule_intent

        prompt = self.router_prompt_template.format(query=query)
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()
            
            if "CHITCHAT" in answer:
                return QueryIntent.CHITCHAT
            return QueryIntent.RAG
            
        except Exception as e:
            print(f"[Router Warning] Lỗi khi gọi LLM phân loại: {e}. Fallback về RAG.")
            # Default an toàn luôn là RAG nếu LLM router gặp sự cố (timeout, rate limit)
            return QueryIntent.RAG

