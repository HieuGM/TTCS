import json
import os
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever

from .indexer import get_indexer
from .config import GOOGLE_API_KEY

# -------------------------------------------------------------
# 1. Custom Ensemble Retriever (Reciprocal Rank Fusion - RRF)
# -------------------------------------------------------------
def get_base_retriever(k=20):
    vectorStore, _ = get_indexer()
    return vectorStore.as_retriever(search_kwargs={"k": k})

def get_bm25_retriever(k=20):
    file_path = r"E:\TTCS\person3_resolved.jsonl"
    documents = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                source_doc = data.get("source_doc", "")
                article = data.get("article", "")
                clause = data.get("clause", "") or ""
                point = data.get("point", "") or ""
                raw_text = data.get("text", "")
                
                location = ", ".join(filter(None, [article, clause, point]))
                page_content = f"{source_doc} — {location}:\n{raw_text}"
                
                metadata = {key: val for key, val in data.items() if key != "text"}
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
    
    if documents:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        return bm25_retriever
    return None

def reciprocal_rank_fusion(doc_lists: List[List[Document]], c=60) -> List[Document]:
    """Kết hợp danh sách document từ nhiều nguồn (BM25, Vector) bằng RRF"""
    fused_scores = {}
    doc_map = {}
    
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            doc_id = doc.page_content  # Dùng page_content làm unique ID
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (rank + c)
            
    # Sắp xếp theo RRF score giảm dần
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, v in reranked]

def ensemble_retrieve(query: str, vector_retriever, bm25_retriever) -> List[Document]:
    """Gọi cả 2 retriever và trộn kết quả bằng RRF"""
    vector_docs = vector_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
    
    # Kết hợp bằng RRF
    return reciprocal_rank_fusion([vector_docs, bm25_docs])

# -------------------------------------------------------------
# 2. Custom Multi-Query Generator
# -------------------------------------------------------------
def generate_queries(query: str, llm) -> List[str]:
    """Sử dụng LLM để sinh 3 câu hỏi biến thể"""
    prompt = f"""Bạn là một trợ lý pháp lý AI. Người dùng đang tìm kiếm thông tin về pháp luật Việt Nam.
Nhiệm vụ của bạn là tạo ra đúng 3 câu hỏi biến thể khác nhau từ câu hỏi gốc của người dùng.
Hãy sử dụng từ đồng nghĩa, cấu trúc lại câu hoặc mở rộng các khái niệm liên quan để giúp hệ thống tìm kiếm tài liệu tốt hơn.
Chỉ trả về 3 câu hỏi (mỗi câu 1 dòng), KHÔNG giải thích gì thêm, KHÔNG dùng dấu gạch đầu dòng hay số thứ tự.

Câu hỏi gốc: {query}"""
    
    try:
        res = llm.invoke(prompt)
        lines = res.content.strip().split('\n')
        
        # Làm sạch kết quả trả về
        variations = []
        for line in lines:
            clean_line = line.strip('-*1234567890. ')
            if clean_line:
                variations.append(clean_line)
                
        # Đảm bảo có câu hỏi gốc trong danh sách
        if query not in variations:
            variations.insert(0, query)
            
        return variations[:4] # Câu gốc + 3 biến thể
    except Exception as e:
        print(f"\n⚠️ Lỗi sinh biến thể (Google API quá tải): {e}")
        print("-> Tạm thời bỏ qua Multi-Query, chỉ dùng câu hỏi gốc để tìm kiếm.")
        return [query]

# -------------------------------------------------------------
# 3. Gemini Reranker
# -------------------------------------------------------------
class GeminiReranker:
    def __init__(self, top_n=5):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        self.top_n = top_n
        self.prompt = PromptTemplate.from_template(
            """Bạn là một hệ thống đánh giá tài liệu pháp lý chuyên nghiệp.
Câu hỏi của người dùng: {query}

Dưới đây là danh sách các tài liệu được trích xuất. Dựa trên câu hỏi, hãy đánh giá độ phù hợp của TỪNG tài liệu (từ 0 đến 10, trong đó 10 là cực kỳ quan trọng và có chứa trực tiếp mức phạt/quy định liên quan, 0 là hoàn toàn không liên quan).

{documents_text}

Bạn PHẢI trả về kết quả dưới dạng một danh sách (mảng) số nguyên tương ứng với thứ tự tài liệu, KHÔNG giải thích gì thêm.
Ví dụ nếu có 3 tài liệu: [10, 0, 5]
KẾT QUẢ CỦA BẠN:"""
        )

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return documents
            
        # Nối tất cả tài liệu thành 1 chuỗi để chấm điểm trong 1 lần gọi API (tránh Rate Limit)
        docs_text = ""
        for i, doc in enumerate(documents):
            # Cắt ngắn mỗi tài liệu để tránh vượt quá token limit
            content = doc.page_content[:1500].replace('\n', ' ')
            docs_text += f"[Tài liệu {i+1}]: {content}\n"
            
        input_prompt = self.prompt.format(query=query, documents_text=docs_text)
            
        try:
            response = self.llm.invoke(input_prompt)
            # Dùng regex để tìm mảng JSON trong câu trả lời
            import re
            match = re.search(r'\[(.*?)\]', response.content)
            if match:
                scores_str = match.group(1).split(',')
                scores = [int(s.strip()) for s in scores_str if s.strip().isdigit()]
            else:
                scores = []
                
            for i, doc in enumerate(documents):
                doc.metadata['relevance_score'] = scores[i] if i < len(scores) else 0
        except Exception as e:
            print(f"Lỗi quá trình rerank: {e}")
            for doc in documents:
                doc.metadata['relevance_score'] = 0
                
        documents.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        return documents[:self.top_n]

# -------------------------------------------------------------
# 4. Pipeline Chính (Retrieve & Rerank)
# -------------------------------------------------------------
def retrieve_and_rerank(query: str, top_k=20, top_n=5) -> List[Document]:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)
    
    print(f"\n========== [BƯỚC 1] TẠO MULTI-QUERY ==========")
    queries = generate_queries(query, llm)
    print("Các biến thể câu hỏi sẽ tìm kiếm:")
    for q in queries:
        print(f" - {q}")

    print(f"\n========== [BƯỚC 2] ENSEMBLE RETRIEVE (VECTOR + BM25) ==========")
    vector_retriever = get_base_retriever(k=top_k)
    bm25_retriever = get_bm25_retriever(k=top_k)
    
    all_docs = []
    seen_contents = set()
    
    # Lặp qua từng câu hỏi biến thể để tìm kiếm
    for q in queries:
        docs = ensemble_retrieve(q, vector_retriever, bm25_retriever)
        # Khử trùng lặp document
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_docs.append(doc)
                
    print(f"\nĐã gộp tổng cộng {len(all_docs)} tài liệu không trùng lặp từ tất cả câu hỏi.")
    
    for i, doc in enumerate(all_docs[:3]):
        print(f"\n[Raw Doc {i+1}]")
        print(f"Nội dung: {doc.page_content[:200]}...")
        
    print(f"\n... Đang chạy Gemini Reranker để chọn ra Top {top_n} tài liệu. Vui lòng đợi ...\n")
    
    # 3. Rerank bằng Gemini
    reranker = GeminiReranker(top_n=top_n)
    final_docs = reranker.compress_documents(all_docs, query)

    print(f"\n========== [BƯỚC 3] TOP {top_n} TÀI LIỆU CUỐI CÙNG ==========")
    for i, doc in enumerate(final_docs):
        score = doc.metadata.get('relevance_score', 'N/A')
        print(f"\n[Reranked Doc {i+1}] | Điểm Gemini chấm: {score}/10")
        print(f"Nội dung: {doc.page_content[:250]}...")
        print(f"Metadata: {doc.metadata}")
    print("\n===========================================================\n")
    
    return final_docs