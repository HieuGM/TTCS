import uuid
from src.generator import ask_legal_bot

def main():
    print("="*50)
    print("HỆ THỐNG LEGAL RAG - PHÁP LUẬT VIỆT NAM (Beta)")
    print("Mô hình nhúng: OpenAI | Tra cứu: MongoDB hybrid + BGE/RRF | Trả lời: DeepSeek V4 Pro")
    print("Gõ 'quit', 'exit' hoặc 'q' để thoát.")
    print("="*50)
    
    # Tạo một session ID ngẫu nhiên cho mỗi phiên chạy CLI để giữ lịch sử hội thoại
    session_id = str(uuid.uuid4())
    print(f"[Session ID: {session_id}]")
    
    while True:
        try:
            print("\n" + "-"*50)
            user_input = input("👤 Bạn: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Tạm biệt!")
                break
                
            if not user_input.strip():
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
