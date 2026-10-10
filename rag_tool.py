### COMPLETE THE CODE  ###

from policy_loader import load_policy_documents

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

from langchain_core.tools import tool
from rag import build_vectorstore, build_retriever

_vectorstore = None
_retriever = None


def _get_retriever():
    global _vectorstore, _retriever
    if _retriever is None:
        _vectorstore = build_vectorstore()
        _retriever = build_retriever(_vectorstore, k=4)
    return _retriever


def search_policy(query: str) -> str:
    retriever = _get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "NO_RELEVANT_POLICY_FOUND"
    parts = []
    for d in docs:
        pid = d.metadata.get("policy_id", "unknown")
        topic = d.metadata.get("topic", "unknown")
        parts.append(f"[{pid} | topic={topic}]\n{d.page_content}")
    return "\n\n".join(parts)


policy_search_tool = tool(
    "search_university_it_policy",
    description=(
        "Search the university IT policy knowledge base. "
        "Use this for any question about IT services, passwords, Wi-Fi, VPN, "
        "printing, storage, or email. Returns relevant policy excerpts."
    ),
)(search_policy)