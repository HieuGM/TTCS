import uuid
from src.generator import ask_legal_bot, get_semantic_cache
from src.pipeline_config import DEFAULT_CACHE_CONFIG

def main():
    print("="*60)
    print("  HỆ THỐNG LEGAL RAG - PHÁP LUẬT VIỆT NAM (Beta)")
    print("  Mô hình nhúng: HuggingFace | Tra cứu: MongoDB hybrid + BGE/RRF | Trả lời: DeepSeek")
    print("="*60)
    
    # Hiển thị trạng thái Semantic Cache
    cfg = DEFAULT_CACHE_CONFIG
    if cfg.enable_semantic_cache:
        print(f"  ⚡ Semantic Cache: BẬT | Threshold: {cfg.cache_similarity_threshold} | TTL: {cfg.cache_ttl_hours}h | Max: {cfg.cache_max_size}")
    else:
        print("  ⚡ Semantic Cache: TẮT")
    
    print("-"*60)
    print("  Lệnh đặc biệt:")
    print("    'quit'/'exit'/'q'  — Thoát")
    print("    'clear-cache'      — Xóa toàn bộ Semantic Cache")
    print("    'cache-stats'      — Xem thống kê cache")
    print("="*60)
    
    # Tạo một session ID ngẫu nhiên cho mỗi phiên chạy CLI để giữ lịch sử hội thoại
    session_id = str(uuid.uuid4())
    print(f"[Session ID: {session_id}]")
    
    while True:
        try:
            print("\n" + "-"*50)
            user_input = input("👤 Bạn: ")
            
            if user_input.lower().strip() in ["quit", "exit", "q"]:
                print("👋 Tạm biệt!")
                break
            
            if not user_input.strip():
                continue
            
            # ── Lệnh clear-cache ──
            if user_input.lower().strip() == "clear-cache":
                cache = get_semantic_cache()
                count = cache.invalidate_all()
                print(f"🗑️  Đã xóa {count} entries từ Semantic Cache.")
                continue
            
            # ── Lệnh cache-stats ──
            if user_input.lower().strip() == "cache-stats":
                cache = get_semantic_cache()
                stats = cache.stats()
                print("\n📊 === THỐNG KÊ SEMANTIC CACHE ===")
                print(f"  Trạng thái:      {'BẬT' if stats['enabled'] else 'TẮT'}")
                print(f"  Entries (memory): {stats['memory_entries']}")
                print(f"  Entries (MongoDB):{stats['db_entries']}")
                print(f"  Tổng cache hits:  {stats['total_hits']}")
                print(f"  Threshold:        {stats['similarity_threshold']}")
                print(f"  TTL:              {stats['ttl_hours']} giờ")
                print(f"  Max size:         {stats['max_size']}")
                print("=" * 36)
                continue
                
            response = ask_legal_bot(session_id, user_input)
            print("\n🤖 Trợ lý Pháp lý:\n")
            print(response)
        
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi hệ thống: {e}")

if __name__ == "__main__":
    main()

