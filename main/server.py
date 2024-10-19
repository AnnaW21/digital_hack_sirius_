from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import openpyxl
import io
from preprocessing import preprocessing, predict_question, get_frequencies, get_dict_of_key_terms
import pandas as pd
import pickle
import csv

app = FastAPI()

# Подключаем статику (для HTML, CSS, JS файлов)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загрузка модели (!!! поменять формат файла с моделью)
model = pickle.load(open('model.pkl', 'rb'))

# Маршрут для отображения HTML-страницы
@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# WebSocket обработчик для получения бинарных данных файла
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Получаем бинарные данные от клиента
            data = await websocket.receive_bytes()
            print(f"Received {len(data)} bytes")

            # Преобразуем полученные байты в объект файла (io.BytesIO)
            file_like = io.BytesIO(data)

            # # Загружаем файл как Excel (openpyxl)
            # workbook = openpyxl.load_workbook(file_like)
            # sheet = workbook.active

            # # Для примера, получаем данные из первой строки
            # first_row = [cell.value for cell in sheet[1]]
            # print("First row data:", first_row)

            # Предобрабатываем данные (пока один столбец!!!!)
            df = preprocessing(file_like)

            # Прописать цикл для прохождения по столбцам датасета (по вопросам)
            # Предсказываем кластеры по первому вопросу
            key_terms = predict_question(df, model)
            print("key_terms", key_terms)

            list_key_terms = pd.Series(get_dict_of_key_terms(key_terms))
            print("list_key_terms", list_key_terms)

            # Выдаем график и текст по первому вопросу
            frequency = pd.Series(get_frequencies(key_terms))

            key_terms_frequency_df = pd.concat([list_key_terms, frequency], axis=1)
            print("key_terms_frequency_df", key_terms_frequency_df)

            # Сериализация DataFrame в формат JSON
            key_terms_frequency_csv = key_terms_frequency_df.to_csv()

            with open('static/data_1.csv', 'w') as f:
                f.write(key_terms_frequency_csv)

            # Отправляем данные JSON через соединение websocket
            # await websocket.send(key_terms_frequency_json)

            # Отправка через WebSocket
            # await websocket.send(key_terms_frequency_json)

            # await websocket.send(key_terms_frequency_df)

            # Отправляем обратно клиенту график и текст по первому вопросу

            # await websocket.send_text(','.join(map(str, first_row)))

    except WebSocketDisconnect:
        print("Client disconnected")
