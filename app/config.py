<<<<<<< HEAD
from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()  # loads variables from .env into environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_HOST = os.getenv("DB_HOST", "db")               # service name in docker-compose
=======
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

# Embedding params
MAX_CHARS = 4000
BATCH_SIZE = 64
SAVE_EVERY = 50

# DB config (for laws/articles migration)
DB_HOST = os.getenv("DB_HOST", "db")
>>>>>>> 53babd6 (hello)
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "legal_ai")
DB_PASSWORD = os.getenv("DB_PASSWORD", "secret")
DB_NAME = os.getenv("DB_NAME", "legal_ai")
<<<<<<< HEAD
def get_connection():
    
=======


def get_connection():
>>>>>>> 53babd6 (hello)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
<<<<<<< HEAD
    )
=======
    )


def get_client() -> OpenAI:
    return OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
>>>>>>> 53babd6 (hello)
