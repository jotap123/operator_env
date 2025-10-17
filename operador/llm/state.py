from enum import Enum
from typing import Optional, List
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class Action(str, Enum):
    SEARCH = "search"
    ERROR = "error"
    NONE = "none"


class State(MessagesState):
    summary: str
    context: Optional[str] = ""
    curated_query: str
    action_plan: Action
    retrieved_docs: List[Document]