from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def retrieve_evidence(query, k=4):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "clinical_agent/vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=k)
    return docs
