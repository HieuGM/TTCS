"""
╔══════════════════════════════════════════════════════════════╗
║  Legal RAG Vietnam — Giao diện Streamlit Premium             ║
║  Hybrid Search MongoDB × BGE Reranker × DeepSeek V4 Pro      ║
╚══════════════════════════════════════════════════════════════╝
"""

import re
import subprocess
import sys
import traceback
import uuid
from pathlib import Path


def _running_in_streamlit() -> bool:
    """Detect whether this script is being executed by the Streamlit runner."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:
            return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _running_in_streamlit():
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.fileWatcherType",
        "none",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command))

import streamlit as st

# ── Cấu hình trang (PHẢI đặt đầu tiên, trước mọi lệnh st khác) ──
st.set_page_config(
    page_title="Legal AI Vietnam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS TÙY CHỈNH — Dark Neon Premium Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --cyan: #00E5FF;
    --gold: #FFD700;
    --dark-bg: #0E1117;
    --card-bg: #161B22;
    --sidebar-bg: #1A1C24;
    --border: #2D303E;
    --text: #E0E0E0;
    --text-muted: #8B949E;
    --success: #3FB950;
    --danger: #F85149;
}

/* ── Global ── */
.stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ── Header glow effect ── */
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -0.5px;
}
.sub-title {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 2px;
    font-weight: 500;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.3rem;
    color: var(--cyan);
}

/* ── Glassmorphism cards ── */
.glass-card {
    background: rgba(22, 27, 34, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(45, 48, 62, 0.6);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-active {
    background: rgba(63, 185, 80, 0.15);
    color: var(--success);
    border: 1px solid rgba(63, 185, 80, 0.3);
}
.badge-inactive {
    background: rgba(248, 81, 73, 0.15);
    color: var(--danger);
    border: 1px solid rgba(248, 81, 73, 0.3);
}

/* ── Stat numbers ── */
.stat-number {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--cyan);
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.25s ease;
    border: 1px solid var(--border);
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0, 229, 255, 0.15);
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 8px;
}

/* ── Thinking expander ── */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.85rem;
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 16px 0;
    border: none;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.7rem;
    padding: 20px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LAZY IMPORTS — Chỉ tải khi thực sự cần thiết
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_backend():
    """
    Tải toàn bộ backend (generator, cache) MỘT LẦN DUY NHẤT.
    Sử dụng st.cache_resource để tránh reload mỗi lần Streamlit rerun script.
    """
    from src.generator import ask_legal_bot, get_semantic_cache
    from src.pipeline_config import DEFAULT_CACHE_CONFIG
    return ask_legal_bot, get_semantic_cache, DEFAULT_CACHE_CONFIG


def safe_load_backend():
    """Wrapper bắt lỗi xung quanh load_backend()."""
    try:
        return load_backend()
    except Exception as e:
        st.error(f"❌ Không thể khởi tạo backend: {e}")
        st.code(traceback.format_exc(), language="text")
        return None, None, None


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Xin chào! 👋 Tôi là **Trợ lý Pháp lý AI** chuyên về Luật Giao thông Việt Nam.\n\n"
                "Bạn có thể hỏi tôi bất kỳ điều gì, ví dụ:\n"
                "- *Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?*\n"
                "- *Vượt đèn đỏ phạt gì?*\n"
                "- *Chở quá số người quy định bị xử lý thế nào?*"
            ),
            "thinking": None,
        }
    ]


# ══════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ══════════════════════════════════════════════════════════════

def parse_thinking(text: str):
    """
    Bóc tách khối <thinking>...</thinking> từ response của DeepSeek.
    Returns: (thinking_content | None, main_content)
    """
    if not text:
        return None, text or ""
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        main_content = text.replace(match.group(0), "").strip()
        return thinking_content, main_content
    return None, text


def get_cache_stats_safe(get_semantic_cache_fn, config):
    """Lấy thống kê cache an toàn, trả về dict mặc định nếu lỗi."""
    default = {
        "enabled": False, "total_hits": 0, "db_entries": 0,
        "memory_entries": 0, "similarity_threshold": "N/A",
        "ttl_hours": "N/A", "max_size": "N/A",
    }
    if get_semantic_cache_fn is None or config is None:
        return default
    if not config.enable_semantic_cache:
        return default
    try:
        cache = get_semantic_cache_fn()
        return cache.stats()
    except Exception:
        return default


def render_router_notice(message: str | None, kind: str = "info", target=None) -> None:
    """Hiển thị trạng thái phân luồng câu hỏi trong UI."""
    if not message:
        return
    target = target or st
    if kind == "rag":
        target.success(message)
    elif kind in {"warning", "fallback"}:
        target.warning(message)
    else:
        target.info(message)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

ask_legal_bot_fn, get_semantic_cache_fn, cache_config = safe_load_backend()

with st.sidebar:
    # Logo + Title
    st.markdown('<h1>⚖️ Legal AI</h1>', unsafe_allow_html=True)
    st.caption("Hệ thống Tư vấn Pháp luật Giao thông")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Nút hội thoại mới ──
    if st.button("✨ Hội thoại mới", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Đoạn hội thoại mới! Hãy hỏi tôi bất kỳ điều gì về luật giao thông.",
                "thinking": None,
            }
        ]
        st.rerun()

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Semantic Cache Panel ──
    st.markdown("##### ⚡ Semantic Cache")
    stats = get_cache_stats_safe(get_semantic_cache_fn, cache_config)

    if stats["enabled"]:
        badge_html = '<span class="badge badge-active">● Hoạt động</span>'
    else:
        badge_html = '<span class="badge badge-inactive">● Đã tắt</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    if stats["enabled"]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-number">{stats["total_hits"]}</div>'
                f'<div class="stat-label">Cache Hits</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-number">{stats["db_entries"]}</div>'
                f'<div class="stat-label">Bản ghi</div>',
                unsafe_allow_html=True,
            )

        st.caption(f"Ngưỡng: **{stats['similarity_threshold']}** · TTL: **{stats['ttl_hours']}h**")

        if st.button("🗑️ Xóa Cache", use_container_width=True):
            try:
                cache = get_semantic_cache_fn()
                count = cache.invalidate_all()
                st.success(f"Đã xóa {count} bản ghi!")
            except Exception as exc:
                st.error(f"Lỗi: {exc}")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Thông tin hệ thống ──
    with st.expander("🔧 Thông tin hệ thống", expanded=False):
        st.caption(f"**Session:** `{st.session_state.session_id[:8]}...`")
        st.caption("**Embedding:** vietnamese-bi-encoder")
        st.caption("**Reranker:** BGE-v2-m3 (Remote)")
        st.caption("**LLM:** DeepSeek V4 Pro")
        st.caption("**Search:** Hybrid (Dense + BM25)")


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════

# Header
st.markdown('<div class="main-title">⚖️ Legal RAG Vietnam</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hybrid Search MongoDB × BGE Reranker × DeepSeek V4 Pro</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# ── Hiển thị lịch sử chat ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_router_notice(msg.get("router_notice"), msg.get("route_kind", "info"))
        if msg.get("thinking"):
            with st.expander("🧠 Quy trình tư duy (Chain-of-Thought)", expanded=False):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

# ── Xử lý câu hỏi mới ──
if prompt := st.chat_input("Nhập câu hỏi pháp lý (VD: Vượt đèn đỏ bị phạt bao nhiêu?)"):

    # Hiển thị câu hỏi user
    st.session_state.messages.append({"role": "user", "content": prompt, "thinking": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý trả lời
    with st.chat_message("assistant"):
        if ask_legal_bot_fn is None:
            st.error("❌ Backend chưa sẵn sàng. Vui lòng kiểm tra cấu hình .env và khởi động lại.")
        else:
            router_state = {"message": None, "kind": "info"}
            router_notice_box = st.empty()

            def update_router_notice(message: str, kind: str = "info") -> None:
                router_state["message"] = message
                router_state["kind"] = kind
                render_router_notice(message, kind, router_notice_box)

            with st.spinner("⏳ Đang tra cứu & phân tích..."):
                try:
                    raw_response = ask_legal_bot_fn(
                        st.session_state.session_id,
                        prompt,
                        status_callback=update_router_notice,
                    )
                    thinking, main_answer = parse_thinking(raw_response)

                    if thinking:
                        with st.expander("🧠 Quy trình tư duy (Chain-of-Thought)", expanded=False):
                            st.markdown(thinking)

                    st.markdown(main_answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": main_answer,
                        "thinking": thinking,
                        "router_notice": router_state["message"],
                        "route_kind": router_state["kind"],
                    })
                except Exception as e:
                    error_msg = f"❌ Lỗi: {str(e)}"
                    st.error(error_msg)
                    with st.expander("📋 Chi tiết lỗi"):
                        st.code(traceback.format_exc(), language="text")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "thinking": None,
                        "router_notice": router_state["message"],
                        "route_kind": router_state["kind"],
                    })

# Footer
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Legal RAG Vietnam v2.0 · TTCS Project · Powered by DeepSeek & MongoDB Atlas</div>',
    unsafe_allow_html=True,
)
