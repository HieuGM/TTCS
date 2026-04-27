import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
from .config import MONGODB_URI, DB_NAME, COLLECTION_NAME, OPENAI_API_KEY

def get_indexer():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    
    vectorStore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index", # Tên index đã được định nghĩa trên Atlas
        relevance_score_fn="cosine",
        text_key="text",
        embedding_key="embedding"
    )
    return vectorStore, collection

def ingest_data(file_path: str):
    """
    Đọc dữ liệu từ file jsonl và index vào MongoDB.
    """
    vectorStore, collection = get_indexer()
    
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Theo cấu trúc JSON thực tế của file pháp luật Việt Nam
            source_doc = data.get("source_doc", "")
            title     = data.get("title", "")
            raw_text  = data.get("text", "")
            article   = data.get("article", "")
            clause    = data.get("clause", "") or ""
            point     = data.get("point", "") or ""

            # --- Tối ưu Embedding cho Legal RAG ---
            # raw_text đã có header "[Điều X. ...]" → KHÔNG cần lặp title
            # Chỉ cần thêm: (1) tên nghị định (user hay hỏi "Nghị định 168..."),
            #               (2) vị trí Điều/Khoản/Điểm để phân biệt giữa các khoản
            # Dùng natural language (không dùng "Văn bản:", "Phần:") → embedding tốt hơn
            location = ", ".join(filter(None, [article, clause, point]))
            page_content = (
                f"{source_doc} — {location}:\n"
                f"{raw_text}"
            )

            # Metadata: dùng toàn bộ data nhưng loại bỏ field 'text' để tránh nhân đôi
            # (page_content đã chứa toàn bộ nội dung text rồi)
            metadata = {k: v for k, v in data.items() if k != "text"}

            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
    print(f"Bắt đầu index {len(documents)} văn bản vào MongoDB...")
    
    
    
    import time
    batch_size = 200 # Tăng batch size để tối ưu tốc độ, vẫn an toàn với Rate Limit của OpenAI
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Đang xử lý lô văn bản từ {i} đến {i+len(batch)}...")
        vectorStore.add_documents(batch)
        
        if i + batch_size < len(documents):
            print("Đợi 5 giây để làm mát Rate Limit...")
            time.sleep(5)
            
    print("Hoàn tất lập chỉ mục toàn thư!")

if __name__ == "__main__":
    import sys

    # Trỏ đến file đã được resolve cross-reference (bộ dữ liệu chất lượng cao)
    file_target = r"E:\TTCS\person3_resolved.jsonl"

    if not os.path.exists(file_target):
        print(f"Không tìm thấy file: {file_target}")
    else:
        print(f"\\n--- Đang Index file: {os.path.basename(file_target)} ---")
        ingest_data(file_target)