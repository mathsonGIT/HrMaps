import json
from pdfminer.high_level import extract_text
import spacy
from spacy.matcher import Matcher
import re
import glob
import os
from openai import OpenAI
from duckduckgo_search import DDGS
import ast
import re
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


NLP = spacy.load("ru_core_news_md")



AI_Client = OpenAI(
    api_key=st.secrets['OPENAI_KEY'], 
    base_url=st.secrets['OPENAI_URL']
)


def generate_answer_DDGS(question, prompt):
    result = DDGS().chat(keywords = f'{prompt}. {question}' , model = 'gpt-4o-mini', timeout = 20)
    return(result)

def generate_answer_openAI(question, prompt):
    messages = []
    messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": question})
    response = AI_Client.chat.completions.create(
    model="openai/gpt-3.5-turbo", # id модели из списка моделей - можно использовать OpenAI, Anthropic и пр. меняя только этот параметр
    messages=messages,
    temperature=0.7,
    n=1,
    max_tokens=3000, # максимальное число ВЫХОДНЫХ токенов. Для большинства моделей не должно превышать 4096
    extra_headers={ "X-Title": "HRMAPs" }, )# опционально - передача информация об источнике API-вызова
    rezult = response.choices[0].message.content
    return(rezult)

def generate_answer(question, prompt):
    try:
        #response = generate_answer_openAI(question=question, prompt=prompt)
        response = generate_answer_DDGS(question=question, prompt=prompt)
    except:
        response = generate_answer_openAI(question=question, prompt=prompt)
        #response = generate_answer_DDGS(question=question, prompt=prompt)
    finally:
        return(response)


def get_pdf():
    folder_path = 'data/regs'  # Замените на ваш путь
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    return(pdf_files)


def load_data(filepath):
    """
    Загружает данные из JSON-файла.

    :param filepath: путь к файлу .json
    :return: словарь с данными
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Файл не найден: {filepath}")
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON в файле: {filepath}")
    return None


HR_DESCRIPTION = load_data('data/data.json') # Словарь с данными по категориям

PREPROCESS_PATTERN = re.compile(r'["«»()]')
PDF_PATTERN = re.compile(r'[\n]')

class PreProcess():
    def __init__(self):
        self.patterns = None
    def clean(self, patterns, s):
        self.patterns = patterns
        s = s.strip()
        for pattern in patterns:
            s = re.sub(pattern, r' ', s)
        s = re.sub(r'\s{2+}', ' ', s)
        s = re.sub(r'« ', '«', s)
        s = re.sub(r' »', '»', s)
        return(s)

def read_pdf(filepath: str) -> str : 
    """
    Загружает данные из PDF-файла.

    :param filepath: путь к файлу .json
    :return: str
    """
    text = extract_text(filepath)
    return(PreProcess().clean(patterns=[PDF_PATTERN], s= text))



def match_knowledge(text: str) -> str:
    rezult = []
    matcher = Matcher(NLP.vocab)
    pattern_desc = ['KNOW', 'SKILL', 'EXPER']
    patterns = [
        [{"LEMMA": "знание"}],
        [{"LEMMA": "умение"}],
        [{"LEMMA": "навык"}], 
    ]

    for i, pattern in enumerate(patterns):
        matcher.add(pattern_desc[i], [pattern])

    # Process the text
    doc = NLP(text)
    matches = matcher(doc)

    # Use a set to track unique sentences
    unique_sentences = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        sentence = span.sent.text.strip()
        unique_sentences.add(sentence)

    # Return unique sentences as a single string
    return ' '.join(unique_sentences)

def dict_to_text(skills: dict)->str:
    return(' '.join([k for value in skills.values() for k in value]))


def calculate_similarity(text1):
    #rezult = {}
    similarities = {}
    doc1 = NLP(match_knowledge(text1))
    
    
    for key, value in HR_DESCRIPTION.items():
        doc2 = NLP(value)
        # Проверяем, есть ли векторы
        if doc1.vector.size == 0 or doc2.vector.size == 0:
            print(f"Один из документов не имеет векторов для ключа: {key}.")
            continue
        
        # Вычисляем семантическую близость
        similarity = round(doc1.similarity(doc2), 2)
        similarities[key] = similarity
    return similarities


def extract_dict_from_string(input_string):
    # Используем регулярное выражение для извлечения текста между фигурными скобками
    match = re.search(r'\{.*\}', input_string, re.DOTALL)
    if match:
        dict_string = match.group(0)
        # Заменяем одинарные кавычки на двойные
        #dict_string = dict_string.replace("'", '"')
        try:
            # Преобразуем строку в словарь
            result_dict = ast.literal_eval(dict_string)
            return result_dict
        except (ValueError, SyntaxError) as e:
            print(f"Ошибка при преобразовании строки в словарь: {e}")
            return None
    else:
        print("Не удалось найти словарь в строке.")
        return None

def plot_matching_results(data):
    df = pd.DataFrame(list(data.items()), columns=['Профиль', 'Соответствие'])
    colors = ['indigo', 'orange', 'green', 'blue', 'red']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Профиль'],
        y=df['Соответствие'],
        marker_color=colors
        ))

    fig.update_layout(
        title='Соответствие профилей ',
        xaxis_title='Профиль',
        yaxis_title='Соответствие',
        yaxis=dict(range=[0.3, 1]),  # Устанавливаем диапазон Y от 0 до 1
        template='plotly_white'
    )
    return(fig)




    
