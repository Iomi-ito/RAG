import os

API_KEY='<КЛЮЧ>'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")

JSON_PATH = os.path.join(BASE_DIR, "data", "questions.json")
COMPANIES_PATH = os.path.join(BASE_DIR, "data", "companies.json")
INDEX_PATH = os.path.join(BASE_DIR, "data", "rag_index")



