from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")

if __name__ == "__main__":
    #1)Загрузка файлов
    loader = DirectoryLoader(
        PDF_FOLDER,                    
        glob="**/*.pdf",            
        loader_cls=PyPDFLoader,     
    )
    docs = loader.load()

    # 2)Разбиение на чанки 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} sub-documents.")

    # 3)Список компаний
    companies_path = os.path.join(BASE_DIR, "data", "companies.json")
    with open(companies_path, "r", encoding="utf-8") as f:
        all_companies = json.load(f)

    # 4)Добавление компаний в метаданные 
    filtered_chunks = []

    for doc in all_splits:
        chunk_text = doc.page_content.lower()  
        companies_in_chunk = []

        for company in all_companies:
            if company.lower() in chunk_text:
                companies_in_chunk.append(company)

        if companies_in_chunk:
            doc.metadata["companies"] = companies_in_chunk
            filtered_chunks.append(doc)

    # 5.Создание vector-store и сохранение
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(all_splits, embeddings)

    save_folder = os.path.join(BASE_DIR, "rag_index")
    vector_store.save_local(save_folder)

