from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from typing import Callable, Optional

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
#

from langchain_core.messages import HumanMessage, AIMessage


def rewrite_question(
    question: str,
    messages: list,
    llm,
) -> str:
    """
    Viết lại câu hỏi dựa trên lịch sử hội thoại.
    Nếu câu hỏi đã đầy đủ ngữ nghĩa thì giữ nguyên.
    """

    if not messages:
        return question

    history = []

    for msg in messages[-10:]:
        if isinstance(msg, HumanMessage):
            history.append(f"Người dùng: {msg.content}")
        elif isinstance(msg, AIMessage):
            history.append(f"Trợ lý: {msg.content}")

    history_text = "\n".join(history)

    prompt = f"""
Bạn là bộ viết lại câu hỏi cho hệ thống Legal RAG.

Lịch sử hội thoại:
{history_text}

Câu hỏi hiện tại:
{question}

Nhiệm vụ:

- Nếu câu hỏi hiện tại phụ thuộc ngữ cảnh trước đó,
hãy viết lại thành một câu hỏi độc lập và đầy đủ.

- Nếu câu hỏi hiện tại đã đầy đủ ngữ nghĩa và không phụ thuộc vào ngữ cảnh trước đó,
giữ nguyên.

- Không trả lời câu hỏi.

- Không giải thích.

- Chỉ xuất ra duy nhất câu hỏi cuối cùng.

Ví dụ:

Lịch sử:
Người dùng: Tôi đi xe máy không đội mũ bảo hiểm thì bị gì?

Câu hỏi:
Nếu tôi gây tai nạn thì sao?

Kết quả:
Nếu người điều khiển xe máy, xe mô tô không đội mũ bảo hiểm gây tai nạn giao thông thì bị xử phạt như thế nào?
"""

    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()

        if rewritten:
            return rewritten

    except Exception:
        pass

    return question

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

StatusCallback = Callable[[str, str], None]


def _notify_status(
    status_callback: Optional[StatusCallback],
    message: str,
    kind: str = "info",
) -> None:
    if status_callback is None:
        return
    try:
        status_callback(message, kind)
    except TypeError:
        status_callback(message)  # type: ignore[misc]
    except Exception:
        pass


def ask_legal_bot(
    session_id: str,
    question: str,
    status_callback: Optional[StatusCallback] = None,
):
    print("...Đang phân tích ý định câu hỏi (Semantic Routing)...")
    _notify_status(
        status_callback,
        "Router đang phân tích ý định câu hỏi...",
        "routing",
    )
    intent = query_router.route(question)
    
    if intent == QueryIntent.CHITCHAT:
        router_message = "Router: Phát hiện đối thoại thông thường, bỏ qua truy xuất RAG."
        print(f"...[{router_message}]...")
        _notify_status(status_callback, router_message, "chitchat")
        context_str = "HỘI THOẠI THƯỜNG NHẬT"
        print("...Đang tư duy trả lời...")
    else:
        router_message = "Router: Phát hiện câu hỏi pháp lý, kích hoạt truy xuất RAG."
        print(f"...[{router_message}]...")
        _notify_status(status_callback, router_message, "rag")

        history_store = get_session_history(session_id)

        rewritten_question = rewrite_question(
            question,
            history_store.messages,
            llm,
        )
        print("Original :", question)
        print("Rewritten:", rewritten_question)
        # ── SEMANTIC CACHE: Kiểm tra cache trước khi chạy RAG ──
        cache = get_semantic_cache()
        if cache.enabled:
            print("...Đang kiểm tra Semantic Cache...")
            _notify_status(status_callback, "Đang kiểm tra Semantic Cache...", "cache")
            cached_response = cache.lookup(rewritten_question)
            if cached_response is not None:
                cache_stats = cache.stats()
                print(f"\n⚡ CACHE HIT — Trả lời từ Semantic Cache!")
                print(f"   📊 Cache: {cache_stats['memory_entries']} entries | "
                      f"Tổng hits: {cache_stats['total_hits']} | "
                      f"Threshold: {cache_stats['similarity_threshold']}")
                _notify_status(
                    status_callback,
                    "Semantic Cache: HIT - dùng lại câu trả lời đã lưu.",
                    "cache_hit",
                )
                return cached_response
            print("⚡ CACHE MISS — Không có câu trả lời phù hợp trong Semantic Cache.")
            _notify_status(
                status_callback,
                "Semantic Cache: MISS - chuyển sang truy xuất tài liệu.",
                "cache_miss",
            )
        else:
            print("⚡ Semantic Cache: TẮT — Bỏ qua kiểm tra cache.")
            _notify_status(status_callback, "Semantic Cache đang tắt, bỏ qua cache.", "cache_disabled")

        # ── Cache MISS → Chạy full RAG pipeline ──
        print("...Đang truy xuất và đánh giá lại (Reranking) tài liệu...")
        _notify_status(status_callback, "Bắt đầu pipeline truy xuất tài liệu pháp lý...", "pipeline")
        docs = retrieve_and_rerank(rewritten_question, status_callback=status_callback)
        result_lines = ["\n🏆 === TOP 7 KẾT QUẢ TRẢ VỀ TỪ CƠ SỞ DỮ LIỆU ==="]
        for i, doc in enumerate(docs[:7]):
            layer = doc.metadata.get("legal_layer", "N/A")
            subj = doc.metadata.get("subjects", [])
            tops = doc.metadata.get("topics", [])
            rerank_source = doc.metadata.get("rerank_source", "rrf_only")
            bge_rank = doc.metadata.get("bge_rank", "N/A")
            bge_score = doc.metadata.get("bge_score")
            score_text = f"{float(bge_score):.4f}" if bge_score is not None else "N/A"
            preview = doc.page_content.replace("\n", " ")[:120] + "..."
            result_lines.append(
                f"[{i+1}] {layer.upper()} | Xe: {subj} | Chủ đề: {tops} | "
                f"Rerank: {rerank_source} | Rank: {bge_rank} | Score: {score_text}"
            )
            result_lines.append(f"    📝 {preview}")
        result_lines.append("================================================\n")
        print("\n".join(result_lines))
        context_str = format_context(docs)
        print("...Đang tư duy luật (CoT & Reflection)...")
        _notify_status(
            status_callback,
            f"Đã chuẩn bị {len(docs)} chunks ngữ cảnh, đang gọi LLM tạo câu trả lời...",
            "answer_generation",
        )

    if intent == QueryIntent.CHITCHAT:
        _notify_status(status_callback, "Đang gọi LLM trả lời hội thoại thường nhật...", "answer_generation")

    response = chain_with_history.invoke(
        {"question": question, "context": context_str},
        config={"configurable": {"session_id": session_id}}
    )
    
    result = response.content

    # ── SEMANTIC CACHE: Lưu kết quả RAG vào cache ──
    if intent != QueryIntent.CHITCHAT:
        cache = get_semantic_cache()
        if cache.enabled:
            cache.store(rewritten_question, result)
            print("💾 Đã lưu câu trả lời vào Semantic Cache.")
            _notify_status(status_callback, "Đã lưu câu trả lời mới vào Semantic Cache.", "cache_store")

    return result



