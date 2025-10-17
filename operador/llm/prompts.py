ROUTER_PROMPT = """
    You are an expert at routing user queries. Based on the user's latest message and the conversation history,
    determine if you should use a search tool to find real-time or external information.
    Guidelines:
    - SEARCH: If the query requires current events, specific facts, or information beyond your general knowledge.
    - NONE: If the query is conversational, general knowledge, creative, or answerable from your training data.
    - ERROR: If the query is ambiguous, malicious, or nonsensical.

    Conversation History Summary: {summary}
    Return only a single word: SEARCH, NONE, or ERROR.
"""

QUERY_CURATOR_PROMPT = """
    Optimize the following user query for an information retrieval system.
    Extract key entities, concepts, and intent. Remove conversational filler.

    Original Query: {query}
"""

QUERY_EXPANDER_PROMPT = """
    Generate 2-3 alternative phrasings for the following user query to improve search results. 
    Focus on synonyms, rephrasing, and different angles. Return each variation on a new line without numbering.

    User Query: {query}
"""

RETRIEVAL_PROMPT = """
    You are an expert assistant providing helpful and accurate answers based on the provided context.
    Guidelines:

    1. Synthesize a comprehensive answer directly from the context.
    2. If the context does not contain the answer, state that clearly. Do not use outside knowledge.
    3. Cite the source of your information when possible, using the document's metadata.
    4. Maintain a helpful, conversational, and professional tone.
    
    Context: {context}
    
    Previous Conversation Summary: {summary}
"""

DIRECT_RESPONSE_PROMPT = """
    You are a helpful AI assistant. Provide a clear, concise, and friendly answer to the
    user's query based on your general knowledge.
"""