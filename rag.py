from typing import Dict, Any
from dotenv import load_dotenv

# Embeddings & DB
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma

# Modern RAG helpers
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate

# Memory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage

# ---------- RAG Configuration ----------

SYSTEM_PROMPT = """
YOU are a helpful assistant.
Answer ONLY from the provided documents context.
If the context is insufficient, just say you don't know.
You may reframe the answer but cannot alter the information.
"""

combined_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nDOCUMENTS:\n{context}"),
    ("human", "{input}")
])


# ---------- MAIN SERVICE CLASS ----------

class RAGService:
    def __init__(
        self,
        persist_directory: str = "vectordb",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 4
    ):
        load_dotenv()

        # Vector DB
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectordb = Chroma(
            collection_name="my_collection",
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        base_retriever = self.vectordb.as_retriever(search_kwargs={"k": k})

        # ✅ FINAL ANSWERING LLM WITH STREAMING ENABLED
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                task="conversational",
                max_new_tokens=300,
                temperature=0.2,
                streaming=True   # ✅ THIS ENABLES STREAMING
            )
        )

        # Summary llm which will create summary

        self.summary_llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
            temperature=0.1,
                )
            )

        # Question rewriting model (no streaming needed here)
        rewrite_llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="conversational",
            max_new_tokens=150,
            temperature=0.2,
        )

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite last query into a standalone meaningful question."),
            ("human", "{input}")
        ])

        self.history_aware_retriever = create_history_aware_retriever(
            llm=rewrite_llm,
            retriever=base_retriever,
            prompt=rewrite_prompt
        )

        # Context + Answer Formatting
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=combined_prompt,
        )

        # Main RAG chain
        self.retrieval_chain =( 
            create_retrieval_chain(
            retriever=self.history_aware_retriever,
            combine_docs_chain=combine_docs_chain,
        )
        | RunnablePassthrough.assign(
        output=lambda x: x.get("answer", ""))
        )
        
       
        # Session-based memory
        self._session_store = {}
        self.runnable = RunnableWithMessageHistory(
            self.retrieval_chain,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    # Session memory object
    # def get_session_history(self, session_id: str):
    #     if session_id not in self._session_store:
    #         self._session_store[session_id] = InMemoryChatMessageHistory()
    #     return self._session_store[session_id]
    
    def get_session_history(self, session_id: str):
        if session_id not in self._session_store:
            self._session_store[session_id] = WindowedSummaryMemory(
                summary_llm=self.summary_llm,
                window_size=6,
                max_summary_words=120
            )
        return self._session_store[session_id]

    # ---------- NORMAL ASK (NON-STREAMING) ----------
    def ask(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        result = self.runnable.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )

        answer = result.get("answer", "")

        docs = result.get("context", [])
        sources = []
        for i, d in enumerate(docs):
            meta = getattr(d, "metadata", {})
            sources.append({
                "source": meta.get("source", f"doc_{i}"),
                "chunk": meta.get("chunk", i),
                "preview": d.page_content[:200].replace("\n", " ")
            })

        return {
            "answer": answer.strip() if isinstance(answer, str) else "",
            "sources": sources,
            "raw": result
        }

    # ✅ ---------- NEW STREAMING ASK ----------
    def ask_stream(self, query: str, session_id: str = "default"):
        for chunk in self.runnable.stream(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        ):
            if isinstance(chunk, dict) and "answer" in chunk:
                yield chunk["answer"]


class WindowedSummaryMemory:
    def __init__(
        self,
        summary_llm,
        window_size: int = 6,
        max_summary_words: int = 120
    ):
        self._chat = InMemoryChatMessageHistory()
        self.summary = ""
        self.window_size = window_size
        self.summary_llm = summary_llm
        self.max_summary_words = max_summary_words

    # 🔑 REQUIRED BY RunnableWithMessageHistory
    @property
    def messages(self):
        msgs = []
        if self.summary:
            msgs.append(
                SystemMessage(content=f"Conversation summary:\n{self.summary}")
            )
        msgs.extend(self._chat.messages)
        return msgs

    # 🔑 REQUIRED BY RunnableWithMessageHistory
    def add_message(self, message):
        self._chat.add_message(message)
        self._maybe_summarize()
    
    def add_messages(self, messages):
        for message in messages:
            self.add_message(message)


    def _maybe_summarize(self):
        messages = self._chat.messages

        if len(messages) <= self.window_size:
            return

        old_messages = messages[:-self.window_size]

        text = "\n".join(
            f"{m.type}: {m.content}" for m in old_messages
        )

        prompt = f"""
        Update the existing summary using the conversation below.
        Keep it under {self.max_summary_words} words.

        EXISTING SUMMARY:
        {self.summary}

        CONVERSATION:
        {text}
        """

        self.summary = self.summary_llm.invoke(prompt).content.strip()

        # Keep only recent messages
        self._chat.messages = messages[-self.window_size:]

# from typing import Dict, Any
# from dotenv import load_dotenv
# from pydantic import PrivateAttr

# # Embeddings & DB
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
# from langchain_chroma import Chroma

# # Modern RAG helpers
# from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
# from langchain_classic.chains.retrieval import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.prompts import ChatPromptTemplate

# # Memory
# from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
# from langchain_core.chat_history import InMemoryChatMessageHistory


# # ---------- RAG Configuration ----------

# SYSTEM_PROMPT = """
# YOU are a helpful assistant.
# Answer ONLY from the provided documents context.
# If the context is insufficient, just say you don't know.
# You may reframe the answer but cannot alter the information.
# """

# combined_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT + "\n\nDOCUMENTS:\n{context}"),
#     ("human", "{input}")
# ])


# # ---------- TRIMMED MEMORY CLASS ----------

# class TrimmedChatHistory(InMemoryChatMessageHistory):
#     _max_messages: int = PrivateAttr()

#     def __init__(self, max_messages: int = 8):
#         super().__init__()
#         self._max_messages = max_messages

#     def add_message(self, message):
#         super().add_message(message)

#         # Trim history to last N messages
#         if len(self.messages) > self._max_messages:
#             self.messages = self.messages[-self._max_messages:]


# # ---------- MAIN SERVICE CLASS ----------

# class RAGService:
#     def __init__(
#         self,
#         persist_directory: str = r"C:\Users\acer\OneDrive\Desktop\Programming\RAG\vectordb",
#         embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
#         k: int = 4,
#         memory_limit: int = 8   # ✅ control history window here
#     ):
#         load_dotenv()

#         self.memory_limit = memory_limit

#         # Vector DB
#         embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
#         self.vectordb = Chroma(
#             collection_name="my_collection",
#             persist_directory=persist_directory,
#             embedding_function=embeddings
#         )

#         base_retriever = self.vectordb.as_retriever(search_kwargs={"k": k})

#         # ✅ FINAL ANSWERING LLM WITH STREAMING
#         self.llm = ChatHuggingFace(
#             llm=HuggingFaceEndpoint(
#                 repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#                 task="conversational",
#                 max_new_tokens=300,
#                 temperature=0.2,
#                 streaming=True
#             )
#         )

#         # Question rewriting model
#         rewrite_llm = HuggingFaceEndpoint(
#             repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#             task="conversational",
#             max_new_tokens=150,
#             temperature=0.2,
#         )

#         rewrite_prompt = ChatPromptTemplate.from_messages([
#             ("system", "Rewrite last query into a standalone meaningful question."),
#             ("human", "{input}")
#         ])

#         self.history_aware_retriever = create_history_aware_retriever(
#             llm=rewrite_llm,
#             retriever=base_retriever,
#             prompt=rewrite_prompt
#         )

#         combine_docs_chain = create_stuff_documents_chain(
#             llm=self.llm,
#             prompt=combined_prompt,
#         )

#         self.retrieval_chain = (
#             create_retrieval_chain(
#             retriever=self.history_aware_retriever,
#             combine_docs_chain=combine_docs_chain,
#             )
#             | RunnablePassthrough.assign(output=lambda x: x.get("answer", ""))
#         )

#         # ✅ Session-based trimmed memory
#         self._session_store = {}

#         self.runnable = RunnableWithMessageHistory(
#             self.retrieval_chain,
#             get_session_history=self.get_session_history,
#             input_messages_key="input",
#             history_messages_key="history"
#         )

#     # ✅ Only store last N messages
#     def get_session_history(self, session_id: str):
#         if session_id is None:
#             session_id = "default"

#         if session_id not in self._session_store:
#             self._session_store[session_id] = TrimmedChatHistory(max_messages=8)

#         return self._session_store[session_id]
#     # ---------- NORMAL ASK ----------
#     def ask(self, query: str, session_id: str = "default") -> Dict[str, Any]:
#         result = self.runnable.invoke(
#             {"input": query},
#             config={"configurable": {"session_id": session_id}},
#         )

#         answer = (
#             result.get("answer")
#             or result.get("output")
#             or result.get("output_text")
#             or ""
#         )

#         docs = result.get("context", [])
#         sources = []
#         for i, d in enumerate(docs):
#             meta = getattr(d, "metadata", {})
#             sources.append({
#                 "source": meta.get("source", f"doc_{i}"),
#                 "chunk": meta.get("chunk", i),
#                 "preview": d.page_content[:200].replace("\n", " ")
#             })

#         return {
#             "answer": answer.strip(),
#             "sources": sources,
#             "output": answer.strip(),   # ✅ ADD THIS
#             "raw": result
#         }

#     # ✅ STREAMING VERSION
#     def ask_stream(self, query: str, session_id: str = "default"):
#         last_output = ""

#         for chunk in self.runnable.stream(
#             {"input": query},
#             config={"configurable": {"session_id": session_id}},
#         ):
#             if not isinstance(chunk, dict):
#                 continue

#             current = (
#                 chunk.get("answer")
#                 or chunk.get("output")
#                 or chunk.get("output_text")
#             )

#             if not current:
#                 continue

#             # ✅ Only print the NEW part
#             if current.startswith(last_output):
#                 token = current[len(last_output):]
#             else:
#                 token = current

#             last_output = current

#             if token.strip():
#                 yield token


