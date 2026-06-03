from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from .config import MONGODB_URI, DB_NAME, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .retrieval_pipeline import retrieve_and_rerank
from .router import SemanticRouter, QueryIntent
from .semantic_cache import SemanticCache
from .pipeline_config import DEFAULT_CACHE_CONFIG

# Định nghĩa hệ thống Prompt bao gồm CoT và Reflection

system_prompt_template = """Bạn là Trợ lý Pháp lý AI chuyên nghiệp, hỗ trợ tra cứu hệ thống văn bản pháp luật Việt Nam. 
Bạn có nhiệm vụ giải đáp thắc mắc dựa trên các trích lục văn bản được cung cấp.

CONTEXT PHÁP LÝ (Tổng hợp từ nhiều nguồn):
{context}

[LƯU Ý QUAN TRỌNG TỪ HỆ THỐNG]
Nếu CONTEXT ghi là "HỘI THOẠI THƯỜNG NHẬT", đây là câu giao tiếp của người dùng. Hãy trả lời tự nhiên, thân thiện, ngắn gọn và BỎ QUA toàn bộ phần <thinking> cũng như Cấu trúc phản hồi bên dưới.

<thinking>
1. Phân tích Query: Người dùng đang hỏi về vấn đề gì? (Giao thông, Thuế, Hình sự...?)
2. Định vị nguồn: Trong Context có những văn bản nào? (Nghị định 168, Luật Giao thông, Thông tư X...?) Văn bản nào là nguồn chính cho câu hỏi này?
3. Đối chiếu chi tiết: Tìm đúng Điều, Khoản, Điểm. Lưu ý các điều kiện đi kèm (loại phương tiện, mức độ vi phạm).
4. Kiểm tra ngoại lệ: Có quy định "Trừ trường hợp..." hay "Ngoại trừ..." trong context không?
5. Tổng hợp: Kết hợp mức phạt, hình thức bổ sung và biện pháp khắc phục từ các nguồn liên quan.
</thinking>

Cấu trúc phản hồi:

**Căn cứ pháp lý:**
- Liệt kê các văn bản được sử dụng: [Tên Nghị định/Luật/Thông tư]

**Nội dung tư vấn:**
- **Hành vi vi phạm:** [Mô tả ngắn gọn]
- **Mức xử phạt:** [Chi tiết mức phạt tiền]
- **Hình thức xử phạt bổ sung:** [Tước quyền sử dụng giấy phép, tịch thu... nếu có]
- **Dẫn chiếu:** [Ghi rõ: Điểm..., Khoản..., Điều... của Văn bản...]

**Độ tin cậy & Lưu ý:**
- [Đánh giá độ khớp thông tin]
- [Lưu ý về hiệu lực văn bản hoặc các tình huống đặc biệt cần kiểm tra thêm nếu context chưa rõ]

LƯU Ý: Tuyệt đối không tự ý gộp mức phạt của hai văn bản khác nhau trừ khi context có quy định rõ về việc áp dụng đồng thời. Nếu không tìm thấy thông tin trong context, hãy báo rõ cho người dùng."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0
)
query_router = SemanticRouter(llm=llm)

chain = prompt | llm

def get_session_history(session_id: str):
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGODB_URI,
        database_name=DB_NAME,
        collection_name="chat_histories"
    )

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ── Semantic Cache Singleton (lazy init) ──────────────────────

_semantic_cache_instance = None

def get_semantic_cache() -> SemanticCache:
    """Lấy hoặc khởi tạo Semantic Cache singleton."""
    global _semantic_cache_instance
    if _semantic_cache_instance is None:
        cfg = DEFAULT_CACHE_CONFIG
        _semantic_cache_instance = SemanticCache(
            similarity_threshold=cfg.cache_similarity_threshold,
            ttl_hours=cfg.cache_ttl_hours,
            max_size=cfg.cache_max_size,
            collection_name=cfg.cache_collection_name,
            enabled=cfg.enable_semantic_cache,
        )
    return _semantic_cache_instance

# ── Helpers ───────────────────────────────────────────────────

def format_context(documents):
    blocks = []
    for doc in documents:
        raw_status = doc.metadata.get("status", "current")

        blocks.append(f"[Trạng thái hiệu lực: {raw_status}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)

def ask_legal_bot(session_id: str, question: str):
    print("...Đang phân tích ý định câu hỏi (Semantic Routing)...")
    intent = query_router.route(question)
    
    if intent == QueryIntent.CHITCHAT:
        print("...[Router] Phát hiện hội thoại thường nhật (Bỏ qua RAG)...")
        context_str = "HỘI THOẠI THƯỜNG NHẬT"
        print("...Đang tư duy trả lời...")
    else:
        # ── SEMANTIC CACHE: Kiểm tra cache trước khi chạy RAG ──
        cache = get_semantic_cache()
        if cache.enabled:
            print("...Đang kiểm tra Semantic Cache...")
            cached_response = cache.lookup(question)
            if cached_response is not None:
                cache_stats = cache.stats()
                print(f"\n⚡ CACHE HIT — Trả lời từ Semantic Cache!")
                print(f"   📊 Cache: {cache_stats['memory_entries']} entries | "
                      f"Tổng hits: {cache_stats['total_hits']} | "
                      f"Threshold: {cache_stats['similarity_threshold']}")
                return cached_response

        # ── Cache MISS → Chạy full RAG pipeline ──
        print("...Đang truy xuất và đánh giá lại (Reranking) tài liệu...")
        docs = retrieve_and_rerank(question)
        print("\n🏆 === TOP 7 KẾT QUẢ TRẢ VỀ TỪ CƠ SỞ DỮ LIỆU ===")
        for i, doc in enumerate(docs[:7]):
            layer = doc.metadata.get("legal_layer", "N/A")
            subj = doc.metadata.get("subjects", [])
            tops = doc.metadata.get("topics", [])
            preview = doc.page_content.replace("\n", " ")[:120] + "..."
            print(f"[{i+1}] {layer.upper()} | Xe: {subj} | Chủ đề: {tops}")
            print(f"    📝 {preview}")
        print("================================================\n")
        context_str = format_context(docs)
        print("...Đang tư duy luật (CoT & Reflection)...")

    response = chain_with_history.invoke(
        {"question": question, "context": context_str},
        config={"configurable": {"session_id": session_id}}
    )
    
    result = response.content

    # ── SEMANTIC CACHE: Lưu kết quả RAG vào cache ──
    if intent != QueryIntent.CHITCHAT:
        cache = get_semantic_cache()
        if cache.enabled:
            cache.store(question, result)
            print("💾 Đã lưu câu trả lời vào Semantic Cache.")

    return result




