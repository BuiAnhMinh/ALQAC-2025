from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()  # loads variables from .env into environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_HOST = os.getenv("DB_HOST", "db")               # service name in docker-compose
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