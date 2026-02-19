"""Shared constants for the chat_messages persistence layer.

Imported by history, callback, and converters so magic strings live
in one place.
"""

from .models import (
    EXTRA_MODEL_NAME,
    EXTRA_PARENT_RUN_ID,
    EXTRA_RUN_ID,
    EXTRA_TOOL_CALL_ID,
    EXTRA_TOOL_CALLS,
    EXTRA_TOOL_NAME,
    ROLE_AI,
    ROLE_HUMAN,
    ROLE_SYSTEM,
    ROLE_TOOL,
)

# Re-export model-level constants so other modules only need one import.
__all__ = [
    # roles
    "ROLE_AI",
    "ROLE_HUMAN",
    "ROLE_SYSTEM",
    "ROLE_TOOL",
    # extra keys
    "EXTRA_MODEL_NAME",
    "EXTRA_PARENT_RUN_ID",
    "EXTRA_RUN_ID",
    "EXTRA_TOOL_CALL_ID",
    "EXTRA_TOOL_CALLS",
    "EXTRA_TOOL_NAME",
    "EXTRA_CACHE_HIT",
    "EXTRA_QUERY_EMBEDDING",
    # table / columns
    "TABLE_CHAT_MESSAGES",
    "COL_CONVERSATION_ID",
    "COL_MESSAGE_ID",
    "COL_ROLE",
    "COL_CONTENT",
    "COL_EXTRA",
    "COL_CREATED_AT",
    # query params
    "PARAM_CID",
    "PARAM_SYSTEM_ROLE",
    "PARAM_LIM",
    # SQL
    "SQL_SELECT_MESSAGES",
    "SQL_DELETE_MESSAGES",
    # defaults
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_TOOL_NAME",
]

# Table and column names
TABLE_CHAT_MESSAGES = "chat_messages"
COL_CONVERSATION_ID = "conversation_id"
COL_MESSAGE_ID = "message_id"
COL_ROLE = "role"
COL_CONTENT = "content"
COL_EXTRA = "extra"
COL_CREATED_AT = "created_at"

# Query parameter names (bound in SQL as :name)
PARAM_CID = "cid"
PARAM_SYSTEM_ROLE = "system_role"
PARAM_LIM = "lim"

EXTRA_CACHE_HIT = "cache_hit"
EXTRA_QUERY_EMBEDDING = "query_embedding"

# Defaults
DEFAULT_MAX_MESSAGES = 100
DEFAULT_TOOL_NAME = "unknown"

# SQL templates (use with sqlalchemy.text())
SQL_SELECT_MESSAGES = f"""
    SELECT {COL_MESSAGE_ID}, {COL_ROLE}, {COL_CONTENT}, {COL_EXTRA}
    FROM {TABLE_CHAT_MESSAGES}
    WHERE {COL_CONVERSATION_ID} = :{PARAM_CID}
      AND {COL_ROLE} != :{PARAM_SYSTEM_ROLE}
    ORDER BY {COL_CREATED_AT} DESC
    LIMIT :{PARAM_LIM}
"""
SQL_DELETE_MESSAGES = (
    f"DELETE FROM {TABLE_CHAT_MESSAGES} WHERE {COL_CONVERSATION_ID} = :{PARAM_CID}"
)
