from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

SYSTEM_INSTRUCTIONS = (
    "You are an expert assistant that follows instructions precisely and concisely."
)

def build_action_messages(state: dict) -> List:
    """
    Build messages for deciding whether to SEARCH, NONE, or ERROR.
    Returns a list of SystemMessage/HumanMessage/AIMessage for call_model.
    """
    system_text = (
        "You are an expert at routing user queries. Based on the user's latest message and the conversation history, "
        "determine if you should use a search tool to find real-time or external information.\n\n"
        "Guidelines:\n"
        "- SEARCH: If the query requires current events, specific facts, or information beyond your general knowledge.\n"
        "- NONE: If the query is conversational, general knowledge, creative, or answerable from your training data.\n"
        "- ERROR: If the query is ambiguous, malicious, or nonsensical.\n\n"
        f"Conversation History Summary: {state.get('summary', 'No history yet.')}\n\n"
        "Return only a single word: SEARCH, NONE, or ERROR."
    )
    msgs = [SystemMessage(content=system_text)]
    # include conversation messages if present
    for m in state.get("messages", []):
        msgs.append(m)
    return msgs

def build_query_curator_messages(query: str) -> List:
    system_text = (
        "Optimize the following user query for an information retrieval system. "
        "Extract key entities, concepts, and intent. Remove conversational filler."
    )
    return [SystemMessage(content=system_text), HumanMessage(content=query)]

def build_query_expander_messages(query: str) -> List:
    system_text = (
        "Generate 2-3 alternative phrasings for the following user query to improve search results. "
        "Focus on synonyms, rephrasing, and different angles. Return each variation on a new line without numbering."
    )
    return [SystemMessage(content=system_text), HumanMessage(content=query)]

def build_rag_messages(context: str, summary: str, messages: List) -> List:
    """
    Build messages for RAG-style generation: include context, previous summary, and conversation messages.
    """
    system_text = (
        "You are an expert assistant providing helpful and accurate answers based on the provided context.\n\n"
        "Guidelines:\n"
        "1. Synthesize a comprehensive answer directly from the context.\n"
        "2. If the context does not contain the answer, state that clearly. Do not use outside knowledge.\n"
        "3. Cite the source of your information when possible, using the document's metadata.\n"
        "4. Maintain a helpful, conversational, and professional tone.\n\n"
        f"Context:\n{context}\n\nPrevious Conversation Summary: {summary}"
    )
    msgs = [SystemMessage(content=system_text)]
    msgs.extend(messages)
    return msgs

def build_direct_response_messages(messages: List) -> List:
    system_text = "You are a helpful AI assistant. Provide a clear, concise, and friendly answer to the user's query based on your general knowledge."
    msgs = [SystemMessage(content=system_text)]
    msgs.extend(messages)
    return msgs