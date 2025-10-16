from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt to determine the appropriate action (search or direct response)
ACTION_DETERMINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at routing user queries. Based on the user's latest message and the conversation history, determine if you should use a search tool to find real-time or external information.

    Guidelines:
    - SEARCH: If the query requires current events, specific facts, or information beyond your general knowledge. This includes topics that change frequently (e.g., news, stock prices, weather).
    - NONE: If the query is conversational, a general knowledge question, a creative request, or something you can answer from your training data.
    - ERROR: If the query is ambiguous, malicious, or nonsensical.

    Conversation History: {summary}
    Return only a single word: SEARCH, NONE, or ERROR."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Prompt to optimize the user's query for search engines
QUERY_CURATOR_PROMPT = ChatPromptTemplate.from_template(
    """Optimize the following user query for an information retrieval system.
Focus on extracting key entities, concepts, and intent. Remove conversational filler.

Original Query: "{query}"

Optimized Query:"""
)

# Prompt to generate multiple query variations for broader search
QUERY_EXPANDER_PROMPT = ChatPromptTemplate.from_template(
    """Generate 2-3 alternative phrasings for the following user query to improve search results.
Focus on synonyms, rephrasing, and different angles to the core question.
Return each variation on a new line, without numbering.

Original Query: "{query}"
"""
)

# Prompt for the main response generation using retrieved context
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant providing helpful and accurate answers based on the provided context.

Guidelines:
1. Synthesize a comprehensive answer directly from the context.
2. If the context does not contain the answer, state that clearly. Do not use outside knowledge.
3. Cite the source of your information when possible, using the document's metadata.
4. Maintain a helpful, conversational, and professional tone.

Context:
{context}

Previous Conversation Summary: {summary}"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Prompt for generating a response without RAG (direct answer)
DIRECT_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Provide a clear, concise, and friendly answer to the user's query based on your general knowledge."),
    MessagesPlaceholder(variable_name="messages"),
])