import json
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from .indexer import get_indexer
from .config import GOOGLE_API_KEY

def get_base_retriever(k=20):
    vectorStore, _ = get_indexer()
    # Để lọc metadata (Ví dụ: trạng thái: Còn hiệu lực), có thể dùng pre_filter
    # Cấu trúc query phụ thuộc schema của bạn trên MongoDB
    return vectorStore.as_retriever(search_kwargs={"k": k})

class GeminiReranker:
    def __init__(self, top_n=5):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        self.top_n = top_n
        self.prompt = PromptTemplate.from_template(
            """Bạn là một hệ thống đánh giá tài liệu pháp lý chuyên nghiệp.
Câu hỏi của người dùng: {query}
Tài liệu trích xuất: {document_content}

Dựa trên câu hỏi, hãy đánh giá độ phù hợp của tài liệu này (chỉ trả về một số nguyên duy nhất từ 0 đến 10, trong đó 10 là cực kỳ quan trọng và phù hợp để trả lời câu hỏi, 0 là hoàn toàn không liên quan).
ĐIỂM SỐ:"""
        )

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return documents
            
        scored_docs = []
        # Chạy vòng lặp để evaluate từng document (Hoặc có thể dùng LLMChain.batch nếu muốn nhanh)
        for doc in documents:
            try:
                # Ép prompt và lấy điểm
                content = doc.page_content[:2000] # Limit context length để rerank nhanh mượt
                formatted_prompt = self.prompt.format(query=query, document_content=content)
                res = self.llm.invoke(formatted_prompt)
                
                score_str = res.content.strip()
                # Parse lấy số nguyên đầu tiên
                score = int(''.join(filter(str.isdigit, score_str)))
            except Exception as e:
                # Nếu LLM parse lỗi, mặc định cho 0
                score = 0
                
            doc.metadata['relevance_score'] = score
            scored_docs.append(doc)
            
        # Sort by score descending
        scored_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        return scored_docs[:self.top_n]

def retrieve_and_rerank(query: str, top_k=20, top_n=5) -> List[Document]:
    # 1. Hybrid Search qua MongoDB (Base Retriever)
    base_retriever = get_base_retriever(k=top_k)
    base_docs = base_retriever.invoke(query)
    
    # 2. Rerank bằng Gemini
    reranker = GeminiReranker(top_n=top_n)
    final_docs = reranker.compress_documents(base_docs, query)
    
    return final_docs
