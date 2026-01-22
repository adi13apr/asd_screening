import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store(
    pdf_dir="clinical_agent/knowledge_base",
    save_path="clinical_agent/vector_store"
):
    documents = []

    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            print(f"Loading: {path}")
            loader = PDFMinerLoader(path)
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)

    print("✅ Clinical knowledge base indexed successfully (TEXT ONLY)")


if __name__ == "__main__":
    build_vector_store()
