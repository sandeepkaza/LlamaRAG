"""src/generation/prompts.py - RAG prompt templates."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_TEMPLATE = """\
You are a helpful expert assistant. Answer the user's question using ONLY \
the information provided in the context below. If the context doesn't contain \
enough information, say so clearly — do NOT make up facts.

Guidelines:
- Be concise and accurate.
- Cite the source document when possible, e.g. "[Source 2: report.pdf]".
- Use bullet points or numbered lists when helpful.
- If sources conflict, note the discrepancy.

Context:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_TEMPLATE),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the follow-up question as a standalone question. Output ONLY the rewritten question."),
    MessagesPlaceholder("chat_history"),
    ("human", "Follow-up: {question}"),
])

NO_CONTEXT_RESPONSE = (
    "I couldn't find relevant information in the uploaded documents to answer your question. "
    "Please try rephrasing, or upload documents that cover this topic."
)
