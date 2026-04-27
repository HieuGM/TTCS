import os
from dotenv import load_dotenv

# Tải cấu hình từ file .env
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "legal")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_col")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not OPENAI_API_KEY or not GOOGLE_API_KEY:
    print("WARNING: Chưa thiết lập OPENAI_API_KEY hoặc GOOGLE_API_KEY trong .env")
if not COHERE_API_KEY:
    print("WARNING: Chưa thiết lập COHERE_API_KEY trong .env")
