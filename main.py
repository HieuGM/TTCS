import streamlit as st
import uuid
import re

from src.generator import ask_legal_bot, get_semantic_cache
from src.pipeline_config import DEFAULT_CACHE_CONFIG

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Legal AI Vietnam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Tùy chỉnh CSS giao diện Sang trọng (Premium UI) ---
st.markdown("""
<style>
    /* Tổng thể */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Đầu trang */
    h1 {
        color: #00E5FF;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
        font-weight: 800;
    }
    
    /* Căn chỉnh lại sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1C24;
        border-right: 1px solid #2D303E;
    }
    
    /* Nút primary */
    div.stButton > button:first-child {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.2);
    }
    
    /* Expander cho Chain of Thought */
    .streamlit-expanderHeader {
        background-color: #1E212B !important;
        color: #FFD700 !important;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #333;
    }
    
    /* Style cho khung tin nhắn */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #00E5FF;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# --- Khởi tạo Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý Pháp lý AI (phiên bản Premium). Bạn cần tư vấn điều luật giao thông nào hôm nay?", "thinking": None}
    ]

# --- Sidebar: Thông tin & Quản lý Cache ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6090/6090333.png", width=100)
    st.title("⚖️ Bảng Điều Khiển")
    st.markdown("---")
    
    # Nút Reset hội thoại
    if st.button("🔄 Tạo đoạn hội thoại mới", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "assistant", "content": "Đoạn hội thoại mới đã được thiết lập. Hãy hỏi tôi bất kỳ điều gì về luật giao thông!", "thinking": None}
        ]
        st.rerun()
        
    st.markdown("---")
    
    # Thống kê Semantic Cache
    st.subheader("⚡ Semantic Cache")
    cache = get_semantic_cache()
    if DEFAULT_CACHE_CONFIG.enable_semantic_cache:
        stats = cache.stats()
        
        # Tạo bảng hiển thị gọn gàng
        col1, col2 = st.columns(2)
        col1.metric("Tổng Hits", stats['total_hits'])
        col2.metric("Số bản ghi", stats['db_entries'])
        
        st.caption(f"**Trạng thái:** {'🟢 Hoạt động' if stats['enabled'] else '🔴 Đã tắt'}")
        st.caption(f"**Ngưỡng Similarity:** {stats['similarity_threshold']}")
        
        if st.button("🗑️ Xóa toàn bộ Cache", type="primary", use_container_width=True):
            count = cache.invalidate_all()
            st.success(f"Đã dọn dẹp {count} bản ghi trong bộ nhớ!")
            # st.rerun()  # Không cần rerun cứng, Streamlit sẽ tự update
    else:
        st.warning("Semantic Cache đang TẮT trong file .env")

# --- Main App ---
st.title("⚖️ Hệ Thống Legal RAG Vietnam")
st.markdown("*Kiến trúc Hybrid Search MongoDB x BGE Reranker x DeepSeek V4 Pro*")
st.markdown("---")

# Hàm bóc tách nội dung Thinking
def parse_thinking(text):
    """
    Dùng Regex để bóc tách khối <thinking>...</thinking>
    Trả về (thinking_content, main_content)
    """
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        main_content = text.replace(match.group(0), "").strip()
        return thinking_content, main_content
    return None, text

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("🧠 Quy trình tư duy (Chain-of-Thought)"):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

# Xử lý Input người dùng
if prompt := st.chat_input("Nhập câu hỏi pháp lý của bạn vào đây (VD: Lỗi đi ngược chiều phạt bao nhiêu?)..."):
    
    # Lưu và hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": prompt, "thinking": None})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Xử lý và hiển thị trả lời của bot
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu dữ liệu từ MongoDB Atlas và phân tích đa luồng..."):
            try:
                # Gọi hàm lõi của hệ thống
                raw_response = ask_legal_bot(st.session_state.session_id, prompt)
                
                # Bóc tách thinking
                thinking, main_answer = parse_thinking(raw_response)
                
                # Hiển thị phần thinking vào expander
                if thinking:
                    with st.expander("🧠 Quy trình tư duy (Chain-of-Thought)"):
                        st.markdown(thinking)
                        
                # Hiển thị đáp án chính
                st.markdown(main_answer)
                
                # Lưu vào lịch sử
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": main_answer, 
                    "thinking": thinking
                })
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi hệ thống: {str(e)}")
