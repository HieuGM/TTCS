from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from .config import MONGODB_URI, DB_NAME, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .retrieval_pipeline import retrieve_and_rerank

# Định nghĩa hệ thống Prompt bao gồm CoT và Reflection

system_prompt_template = """Bạn là Trợ lý Pháp lý AI chuyên nghiệp, hỗ trợ tra cứu hệ thống văn bản pháp luật Việt Nam. 
Bạn có nhiệm vụ giải đáp thắc mắc dựa trên các trích lục văn bản được cung cấp.

CONTEXT PHÁP LÝ (Tổng hợp từ nhiều nguồn):
{context}

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

def format_context(documents):
    blocks = []
    for doc in documents:
        # Chuyển đổi trạng thái hiệu lực sang tiếng Việt
        raw_status = doc.metadata.get("status", "current")

        
        blocks.append(f"[Trạng thái hiệu lực: {raw_status}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)

def ask_legal_bot(session_id: str, question: str):
    print("...Đang truy xuất và đánh giá lại (Reranking) tài liệu...")
    docs = retrieve_and_rerank(question)
    context_str = format_context(docs)
    
    print("...Đang tư duy luật (CoT & Reflection)...")
    response = chain_with_history.invoke(
        {"question": question, "context": context_str},
        config={"configurable": {"session_id": session_id}}
    )
    
    return response.content
