"""
Persistence / checkpoint selection.
Uses MemorySaver by default, or SqliteSaver / PostgresSaver when configured.
"""
import os
from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer():
    """
    Pick a checkpointer based on environment variables.
    Uses memory when DATABASE_URL is not set.
    """
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("sqlite"):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = db_url.replace("sqlite:///", "")
            return SqliteSaver.from_conn_string(db_path)
        except ImportError:
            pass
    if db_url.startswith("postgresql"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            return PostgresSaver.from_conn_string(db_url)
        except ImportError:
            pass
    return MemorySaver()
