"""
PostgreSQL 会话存储模块
表结构：
  sessions: session_id(PK), title, created_at
  messages: id, session_id(FK→sessions), role, content, created_at
支持：创建/列表/删除会话，存取消息
"""
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List


DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "yang"
DB_PASSWORD = "123456"
DB_NAME = "postgres"



def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, dbname=DB_NAME,
        cursor_factory=RealDictCursor
    )


def init_db():
    """初始化 sessions 和 messages 表（自动调用）"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(64) PRIMARY KEY,
                    title VARCHAR(256) NOT NULL DEFAULT '新会话',
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL
                        REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            conn.commit()


def list_sessions() -> List[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]



def create_session(title: str = "新会话") -> str:
    sid = uuid.uuid4().hex[:16]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, title) VALUES (%s, %s)",
                (sid, title)
            )
            conn.commit()
    return sid


def delete_session(session_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
            conn.commit()


def load_messages(session_id: str) -> List[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM messages WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,)
            )
            return [dict(row) for row in cur.fetchall()]


def save_message(session_id: str, role: str, content: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content)
            )
            conn.commit()


def update_session_title(session_id: str, title: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET title = %s WHERE session_id = %s",
                (title, session_id)
            )
            conn.commit()
