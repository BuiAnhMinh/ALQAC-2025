# config.py
from dotenv import load_dotenv
import os
import psycopg2
from pathlib import Path
from openai import OpenAI

load_dotenv()

# OpenAI / OpenRouter
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
EMB_MODEL = os.getenv("EMB_MODEL", "openai/text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
# Paths
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "alqac25_train.json"
LAW_PATH = DATA_DIR / "alqac25_law.json"
TEST_PATH = DATA_DIR / "alqac25_private_test_Task_1.json"
ZALO_LAW_PATH = DATA_DIR / "zalo_corpus.json"
STOPWORDS_PATH = DATA_DIR / "vietnamese-stopwords.txt"

ARTICLE_EMB_PATH = DATA_DIR / "article_embeddings.npy"
TRAIN_Q_EMB_PATH = DATA_DIR / "train_question_embeddings.npy"
TEST_Q_EMB_PATH = DATA_DIR / "test_question_embeddings.npy"
ARTICLE_TOKENS_PATH = DATA_DIR / "article_tokens.json"

# Embedding params
MAX_CHARS = 4000
BATCH_SIZE = 64
SAVE_EVERY = 50

# DB config (for laws/articles migration)
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "legal_ai")
DB_PASSWORD = os.getenv("DB_PASSWORD", "secret")
DB_NAME = os.getenv("DB_NAME", "legal_ai")


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )


def get_client() -> OpenAI:
    return OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
