import os
from dotenv import load_dotenv

# Tải cấu hình từ file .env
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "legal")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_col")
MONGODB_SEARCH_INDEX = os.getenv("MONGODB_SEARCH_INDEX", "default")
MONGODB_SEARCH_FIELD = os.getenv("MONGODB_SEARCH_FIELD", "text")
MONGODB_TEXT_SEARCH_INDEX = os.getenv("MONGODB_TEXT_SEARCH_INDEX", MONGODB_SEARCH_INDEX)
MONGODB_TEXT_SEARCH_FIELD = os.getenv("MONGODB_TEXT_SEARCH_FIELD", MONGODB_SEARCH_FIELD)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: Chưa thiết lập OPENAI_API_KEY trong .env")
if not COHERE_API_KEY:
    print("WARNING: Chưa thiết lập COHERE_API_KEY trong .env")
if not DEEPSEEK_API_KEY:
    print("WARNING: Chưa thiết lập DEEPSEEK_API_KEY trong .env")
