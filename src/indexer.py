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
        relevance_score_fn="cosine"
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
            title = data.get("title", "")
            raw_text = data.get("text", "")
            
            # Khâu tối ưu cho Embedding: Gắn ngữ cảnh Luật/Tên Điều trước nội dung
            page_content = f"Văn bản: {source_doc}\nPhần: {title}\nNội dung: {raw_text}"
            
            # Dùng toàn bộ data làm metadata để phục vụ Retrieval và Reranker sau này
            metadata = data
            
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
    print(f"Bắt đầu index {len(documents)} văn bản vào MongoDB...")
    
    import time
    batch_size = 50 # Giảm mạnh batch size vì có những đoạn document văn bản rất dài làm vượt 40k token trong TỪNG LÔ
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Đang xử lý lô văn bản từ {i} đến {i+len(batch)}...")
        vectorStore.add_documents(batch)
        
        # Ngủ ngắn vì lô nhỏ - đảm bảo duy trì lượng nạp < 40k Request trong 1 phút
        if i + batch_size < len(documents):
            print("Đợi 20 giây để làm mát Rate Limit (Vượt qua chu kỳ 1 phút của OpenAI)...")
            time.sleep(20)
            
    print("Hoàn tất lập chỉ mục toàn thư!")

if __name__ == "__main__":
    import sys
    import glob
    # path setup
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Tìm toàn bộ các file jsonl vừa được gen ra trong thư mục raw_data
    search_pattern = os.path.join(base_path, "raw_data", "*_structured.jsonl")
    jsonl_files = glob.glob(search_pattern)
    
    if not jsonl_files:
        print("Không tìm thấy file .jsonl nào trong raw_data/")
    else:
        for file_target in jsonl_files:
            print(f"\\n--- Đang Index file: {os.path.basename(file_target)} ---")
            ingest_data(file_target)
