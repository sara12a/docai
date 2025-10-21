

import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings

def extract_pdf_llama(pdf_path, collection_name="pdf_collection", chunk_size=500, chunk_overlap=50, llama_model_path="models/ggml-model-q4_0.bin"):
    """
    Extract text from a PDF, split into chunks using LangChain, embed using LLaMA, and store in Chroma.
    """
    if not os.path.exists(llama_model_path):
        raise ValueError(f"LLaMA model file not found at {llama_model_path}. Please download a local LLaMA model.")

    # 1️⃣ Extract text from PDF
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        if reader.is_encrypted:
            reader.decrypt("")  # adjust if password is needed
        print(f"📄 Total pages: {len(reader.pages)}")

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or "[No text found]"
            text += f"\n\n--- Page {i+1} ---\n\n{page_text}"

    # 2️⃣ Split text into chunks using LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    print(f"🔹 Total chunks created: {len(chunks)}")

    # 3️⃣ Initialize LLaMA embeddings
    embeddings = LlamaCppEmbeddings(model_path=llama_model_path)

    # 4️⃣ Create or load Chroma vector store
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    # 5️⃣ Add documents to Chroma
    metadata_list = [{"source_pdf": os.path.basename(pdf_path), "chunk_index": i} for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=metadata_list)
    vectorstore.persist()  # Save to disk
    print(f"✅ PDF '{os.path.basename(pdf_path)}' stored in Chroma collection '{collection_name}'")

    return vectorstore

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    pdf_file = "data/test2.pdf"  # Your PDF path
    llama_model = "models/ggml-model-q4_0.bin"  # Local LLaMA model path
    extract_pdf_llama(pdf_file, llama_model_path=llama_model)
