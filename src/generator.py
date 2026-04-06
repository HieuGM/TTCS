from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from .config import MONGODB_URI, DB_NAME, GOOGLE_API_KEY
from .retriever import retrieve_and_rerank

# Định nghĩa hệ thống Prompt bao gồm CoT và Reflection
system_prompt_template = """Bạn là Chuyên gia Tư vấn Pháp lý Việt Nam độ chính xác tuyệt đối. Nhiệm vụ của bạn là giải đáp câu hỏi dựa trên lịch sử hội thoại và dữ liệu pháp luật được cung cấp.

CONTEXT PHÁP LÝ (Đã lọc):
{context}

YÊU CẦU DUY NHẤT VÀ BẮT BUỘC:
Bạn PHẢI mô phỏng lại quá trình phân tích pháp y vào thẻ <thinking> trước khi đưa ra câu trả lời cuối cùng.

<thinking>
1. Chain of Thought (Suy luận pháp lý):
- Xác định yêu cầu chính của người dùng.
- Liên kết với các Điều khoản trong Context: Liệt kê Điều, Khoản, Tên văn bản.
- Trạng thái hiệu lực hiện tại là gì? Có văn bản nào hết hiệu lực không?
- Quy định chuyên ngành có ưu tiên quy định chung không?

2. Reflection (Tự đánh giá độ tin cậy):
- Có thông tin nào mâu thuẫn với Hiến pháp/Luật Mẹ không?
- Nếu context không đề cập, bắt buộc phải trả lời: "Tài liệu hiện tại không đề cập". Không chế số hiệu, điều khoản.
</thinking>

CÂU TRẢ LỜI CỦA BẠN DÀNH CHO USER:
(Sau thẻ thinking, viết câu trả lời. Yêu cầu:
- Format markdown rõ ràng, xuống dòng rành mạch.
- Bắt buộc dẫn chiếu rõ [Điều X, Khoản Y, Tên Văn Bản].
- Cảnh báo In đậm nếu liên quan văn bản sắp/đã hết hiệu lực.)"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Sử dụng pro cho khả năng phân tích luật phức tạp (Generation phase)
    google_api_key=GOOGLE_API_KEY,
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
        # Cố gắng lấy trạng thái hiệu lực
        status = doc.metadata.get("status", doc.metadata.get("Trạng thái", "Đang có hiệu lực"))
        blocks.append(f"[Trạng thái hiệu lực: {status}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)

def ask_legal_bot(session_id: str, question: str):
    print("...Đang truy xuất và đánh giá lại (Reranking) tài liệu...")
    docs = retrieve_and_rerank(question, top_k=20, top_n=5)
    context_str = format_context(docs)
    
    print("...Đang tư duy luật (CoT & Reflection)...")
    response = chain_with_history.invoke(
        {"question": question, "context": context_str},
        config={"configurable": {"session_id": session_id}}
    )
    
    return response.content
