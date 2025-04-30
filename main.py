import streamlit as st
from utils import read_pdf, generate_answer, extract_dict_from_string, dict_to_text, calculate_similarity, plot_matching_results
import pandas as pd
# Заголовок приложения
st.sidebar.image("src/main.png", use_container_width=True)  
st.sidebar.title("Карта Кадров - AI Ассистент по подбору студентов")

# Выпадающий список в боковой панели
option = st.sidebar.selectbox("Выберите действие", ["Анализ должностной инструкции", "Подбор кандидата"])

if option == "Анализ должностной инструкции":
    st.title("Анализ должностной инструкции")

    uploaded_file = st.file_uploader("Загрузите PDF файл должностной инструкции", type="pdf")

    if uploaded_file is not None:
        text = read_pdf(uploaded_file)
        hr_prompt = '''
        Ты - профессиональный hr менеджер и подбираешь кандидата для работы в органах государственной власти по должностному регламенту. 
        Тебе будет представлен текст должностной инструкции кандидата. 
        Выдели из текста должностной инструкции только умения и навыки. Выдели только по семь ключевых умений и навыков. Ничего не придумывай,  бери текст только из инструкции. 
        Верни словарь Python с ключами 'УМЕНИЯ' и 'НАВЫКИ'.
        '''
        question = f'Должностная инструкция: {text}'
        
        if st.button("Анализировать"):
            skills = generate_answer(question, hr_prompt)
            skills_dict = extract_dict_from_string(skills)
            match_profession = calculate_similarity(skills)

            with st.container():
                st.title("Результаты анализа должностной инструкции")
                # st.write(match_profession)

                # Отображаем график в Streamlit
                fig = plot_matching_results(match_profession)
                st.plotly_chart(fig)

                col1, col2 = st.columns(2)
                skills_df = pd.DataFrame(skills_dict)
                st.dataframe(
                    skills_df.style
                    .highlight_max(axis=0)  # Подсветка максимальных значений
                    .set_properties(**{'text-align': 'left'})  # Выравнивание текста
                    .set_table_attributes('style="width: 100%; font-size: 12px;"')  # Ширина и размер шрифта
                    )
 
                

elif option == "Подбор кандидата":
    st.title("Подбор кандидата")
    # Здесь можно добавить функционал для подбора кандидата
    st.write("Функционал подбора кандидата будет добавлен позже.")
        
        
