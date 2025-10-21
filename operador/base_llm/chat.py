import logging
import asyncio

from typing import Dict, Any, Optional

from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END

from operador.config import settings
from operador.base_llm.state import Action, State
from operador.base_llm.prompts import (
    ROUTER_PROMPT,
    QUERY_CURATOR_PROMPT,
    QUERY_EXPANDER_PROMPT,
    RETRIEVAL_PROMPT,
    DIRECT_RESPONSE_PROMPT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMAgent:
    def __init__(self):
        self.llm = init_chat_model(
            model_name=settings.LLM_MODEL_NAME,
            model_provider=settings.LLM_PROVIDER,
            region_name=settings.LLM_REGION_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=False,
        )
        self.embeddings = None
        self.search_tool = TavilySearchResults(max_results=settings.MAX_SEARCH_RESULTS)
        self.vectorstore: Optional[Chroma] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.reranker = CrossEncoderReranker(model=settings.CROSS_ENCODER, top_n=5)
        self.memory = MemorySaver()
        self.graph = self.build_graph()


    async def _determine_action(self, state: State) -> State:
        """Determines whether to search or respond directly."""
        chain = ROUTER_PROMPT | self.llm | StrOutputParser()
        try:
            result = await chain.ainvoke({
                "messages": state["messages"],
                "summary": state.get("summary", "No history yet.")
            })
            action = result.strip().upper()
            if "SEARCH" in action:
                state["action_plan"] = "SEARCH"
            elif "NONE" in action:
                state["action_plan"] = "NONE"
            else:
                state["action_plan"] = "ERROR"
        except Exception as e:
            logging.error(f"Action determination failed: {e}")
            state["action_plan"] = "ERROR"
            state["messages"].append(AIMessage(content="Error in processing. Please try again."))
        return state


    async def _curate_and_expand_query(self, state: State) -> State:
        """Optimizes and expands the user's query for better retrieval."""
        original_query = state['messages'][-1].content

        # Curate query
        curate_chain = QUERY_CURATOR_PROMPT | self.llm | StrOutputParser()
        curated_query = await curate_chain.ainvoke({"query": original_query})
        state["curated_query"] = curated_query.strip()

        # Expand query
        expand_chain = QUERY_EXPANDER_PROMPT | self.llm | StrOutputParser()
        expanded_result = await expand_chain.ainvoke({"query": curated_query})
        expanded_queries = [q.strip() for q in expanded_result.split('\n') if q.strip()]
        state["expanded_queries"] = [curated_query] + expanded_queries
        
        return state


    async def load_pdf(self, pdf_path: str):
        """Loads a PDF, splits it into chunks, and sets up the hybrid retriever."""
        try:
            loader = PyPDFLoader(pdf_path)
            docs = await loader.aload()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)

            self.vectorstore = Chroma.from_documents(chunks, self.embeddings)

            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = settings.BM25_K

            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": settings.VECTOR_K}
            )

            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=settings.ENSEMBLE_WEIGHTS
            )
            logging.info(f"Successfully loaded and indexed {pdf_path}")
        except Exception as e:
            logging.error(f"Failed to load PDF {pdf_path}: {e}")
            raise


    async def _retrieve_context(self, state: State) -> State:
        """Retrieves context from documents or the web using hybrid search."""
        if self.ensemble_retriever:
            # Document-based retrieval
            all_docs = []
            # Gather results from all expanded queries concurrently
            results = await asyncio.gather(
                *(self.ensemble_retriever.ainvoke(q) for q in state["expanded_queries"])
            )
            for doc_list in results:
                all_docs.extend(doc_list)

            # Deduplicate documents based on content
            unique_docs = {doc.page_content: doc for doc in all_docs}.values()

            # Rerank the unique documents
            reranked_docs = await self.reranker.acompress_documents(
                documents=unique_docs,
                query=state["curated_query"]
            )

            state["retrieved_docs"] = reranked_docs
            state["context"] = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
        else:
            # Web search fallback
            search_results = await self.search_tool.ainvoke(state["curated_query"])
            state["context"] = "\n\n---\n\n".join([str(res) for res in search_results])

        return state


    async def _generate_response(self, state: State) -> State:
        """Generates a final response based on the action plan and context."""
        if state['action_plan'] == "SEARCH":
            prompt = RETRIEVAL_PROMPT
            inputs = {
                "context": state["context"],
                "summary": state.get("summary", "No history yet."),
                "messages": state['messages']
            }
        else: # "NONE" action
            prompt = DIRECT_RESPONSE_PROMPT
            inputs = {"messages": state['messages']}
        
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke(inputs)
        state["messages"].append(AIMessage(content=response))
        return state


    def _should_continue(self, state: State) -> str:
        """Decides if the conversation history needs summarization."""
        return "summarize" if len(state["messages"]) > settings.MEMORY_SUMMARIZER_THRESHOLD else END


    def summarize_conversation(self, state: State) -> Dict[str, Any]:
        """Enhanced conversation summarization."""
        try:
            current_summary = state.get("summary", "")
            recent_messages = state["messages"][-10:]  # Focus on recent messages

            if current_summary:
                summary_prompt = f"""
                Current summary: {current_summary}
                
                Update this summary by incorporating the key points from the recent conversation
                below. Focus on important facts, decisions, and context that would be valuable
                for future reference.
                """
            else:
                summary_prompt = """Create a concise summary of this conversation, highlighting
                key topics, facts, and important context:"""

            messages_for_summary = recent_messages + [HumanMessage(content=summary_prompt)]
            # synchronous call to call_model (this method is sync)
            response = self.llm.invoke(messages_for_summary)
            
            # Keep only the most recent messages plus the summary
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]
            
            return {
                "summary": response.content.strip(), 
                "messages": delete_messages
            }

        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return {"summary": state.get("summary", ""), "messages": []}


    def build_graph(self) -> CompiledStateGraph:
        """Build the enhanced processing graph."""
        workflow = StateGraph(State)

        workflow.add_node("plan", self._determine_action)
        workflow.add_node("curate_query", self._curate_and_expand_query)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("summarize", self.summarize_conversation)

        workflow.set_entry_point("plan")
        workflow.add_conditional_edges(
            "plan",
            lambda x: x['action_plan'],
            {
                Action.SEARCH: "curate_query",
                Action.NONE: "generate",
                Action.ERROR: END
            }
        )
        workflow.add_edge("curate_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges("generate", self._should_continue)
        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=self.memory)


    async def process_query(self, query: str, thread_id: str) -> str:
        """Processes a user query asynchronously for a given conversation thread."""
        if not query or not query.strip():
            return "Please provide a valid query."
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config
            )
            response_message = final_state["messages"][-1]
            return response_message.content if isinstance(response_message, AIMessage) else "An error occurred."
        except Exception as e:
            logging.error(f"Error processing query for thread {thread_id}: {e}")
            return "I'm sorry, an unexpected error occurred. Please try again."
