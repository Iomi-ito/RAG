import os
import requests
import json
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from config import JSON_PATH, COMPANIES_PATH, INDEX_PATH, API_KEY
from langchain_huggingface import HuggingFaceEmbeddings


def extract_companies_from_question(question_text, all_companies):
    '''Возвращает список компаний, которые есть в вопросе'''
    question = question_text.split("?")[0]
    return [c for c in all_companies if c in question]

def retrieve_docs(vector_store, query, query_companies, top_k = 10):
    '''Ищет документы и фильтрует по компаниям, возвращает список документов'''
    retrieved_docs = vector_store.similarity_search(query, k=20)
    filtered_docs = [
        d for d in retrieved_docs
        if any(c in d.metadata.get("companies", []) for c in query_companies)
    ]
    return filtered_docs[:top_k] if filtered_docs else retrieved_docs[:8]

def build_context(docs):
    '''Объединяет выбранные документы в контекст с нумерацией чанков'''
    return "\n\n".join(f"[CHUNK {i}]\n{d.page_content}" for i, d in enumerate(docs))

def ask_model(prompt, api_key):
    '''Отправка промпта в DeepSeek и получение ответа'''
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

def parse_model_response(raw):
    '''Разбор JSON-ответа модели'''
    answer = "N/A"
    chunk_id = None
    try:
        parsed = json.loads(raw)
        answer = parsed.get("value", "N/A")
        chunk_id_raw = parsed.get("chunk_id", None)
        if chunk_id_raw is not None:
            chunk_id = int(chunk_id_raw)
    except Exception:
        pass
    return answer, chunk_id

def build_references(answer, docs, chunk_id):
    '''Формирует список ссылок на документы'''
    references = []
    if answer != "N/A" and docs and chunk_id is not None and 0 <= chunk_id < len(docs):
        d = docs[chunk_id]
        references.append({
            "pdf_sha1": os.path.splitext(os.path.basename(d.metadata["source"]))[0],
            "page_index": d.metadata.get("page", 0)
        })
    return references

def correct_answer_type(value, kind):
    '''Приведение значения к нужному типу после модели'''
    if kind == "boolean":
        return str(value).lower() == "true"
    elif kind == "number":
        try:
            return float(value)
        except:
            return "N/A"
    else:
        return str(value)

def save_and_submit(submission_data, filename, url, timeout=30):
    '''
    Сохраняет submission в JSON и отправляет. 
    Возвращает: response
    '''
    #сохраняем JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
    
    #отправляем на сервер
    with open(filename, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            timeout=timeout
        )
    
    print(f"Файл {filename} отправлен на {url}")
    print("HTTP status:", response.status_code)
    print("Response text:", response.text)
    
    return response


if __name__ == "__main__":
    #Загрузка вопросов
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    #Загрузка компаний
    with open(COMPANIES_PATH, "r", encoding="utf-8") as f:
        all_companies = json.load(f)

    #Загрузка vector_store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    #Запрос к модели
    model_answers=[] 

    for q in questions:
        question = q["text"].split(sep='?')

        #Выделение компаний:
        query_companies=extract_companies_from_question(q["text"], all_companies)

        #Поиск документов
        docs = retrieve_docs(vector_store, q["text"], query_companies)
    
        context = build_context(docs)

        prompt = f"""
                Ты — ассистент по извлечению данных из документов.

                Ответь на вопрос, используя ТОЛЬКО один фрагмент контекста.

                Верни ответ СТРОГО в JSON формате: {{ "value": ответ, "chunk_id": номер фрагмента }}


                Правила:
                - chunk_id — ОБЯЗАТЕЛЬНО номер фрагмента, из которого взят ответ.
                - Если информация есть — ты ОБЯЗАН выбрать один фрагмент.
                - chunk_id может быть null, только если ответ 'N/A'.
                - N/A разрешено ТОЛЬКО если ни один фрагмент не содержит ответа.
                - Формат value должен строго соответствовать kind: {q["kind"]}.
                - Никакого текста вне JSON.
                Формат ответов: number: Только цифры (прим: 122233, вещественное число - 0.25). Без пробелов, букв, процентов.
                name: Одно имя/наименование.
                names: Несколько имён через запятую и пробел (Name One, Name Two).
                boolean: Только true или false строчными

                Вопрос:
                {q["text"]}

                Контекст:
                {context}
        """
        #Отправка промпта
        raw_answer = ask_model(prompt, API_KEY)

        #Разбор ответа
        answer, chunk_id = parse_model_response(raw_answer)
        references = build_references(answer, docs, chunk_id)
        value_correct = correct_answer_type(answer, q["kind"])

        submission_item = {
            "question_text": q["text"],
            "value": value_correct,
            "references": references
        }
        model_answers.append(submission_item)

    submission = { "team_email": "test@rag-tat.com", "submission_name": "Vinogradova_v2.7", "answers": model_answers}
    save_and_submit(submission, 'submission_Vinogradova_v2.json', "http://5.35.3.130:800/submit", 30)


   
