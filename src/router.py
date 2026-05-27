# File: src/router.py

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

    def route(self, query: str) -> QueryIntent:
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


