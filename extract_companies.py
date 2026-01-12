import spacy
import re
import json
import os
from config import JSON_PATH, COMPANIES_PATH

LEGAL_SUFFIXES = [
    "Inc", "INC", "Corporation", "corp", "CORPORATION", "Corp",
    "Ltd", "ltd.", "plc", "ag", "sa", "group", "Group"
]

#Загрузка модели
nlp = spacy.load("en_core_web_lg")

def extract_orgs(text):
    #Выделение названий организаций из текста
    doc = nlp(text)
    return list({
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG"
    })

def normalize_org(name):
    #Нормализация названий организаций
    name = re.sub(r"[.,\"]", "", name)
    for s in LEGAL_SUFFIXES:
        name = re.sub(rf"\b{s}\b", "", name)

    return re.sub(r"\s+", " ", name).strip()

    
#Выделение компаний и сохранение в json-файл
if __name__ == "__main__":
   
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    all_companies = set()
    for q in questions:
        orgs = extract_orgs(q["text"])
        for org in orgs:
            all_companies.add(normalize_org(org))

    with open(COMPANIES_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(all_companies), f, indent=2, ensure_ascii=False)

    print(f"Сохранено компаний: {len(all_companies)}")