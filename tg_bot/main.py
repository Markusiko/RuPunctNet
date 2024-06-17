import asyncio
import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import sqlite3

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Проверка наличия переменных окружения
BOT_TOKEN = os.environ.get('BOT_TOKEN')
GIGACHAT_TOKEN = os.environ.get('GIGACHAT_TOKEN')

if not BOT_TOKEN:
    raise ValueError("Отсутствует переменная окружения BOT_TOKEN")
if not GIGACHAT_TOKEN:
    raise ValueError("Отсутствует переменная окружения GIGACHAT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Загрузка модели Hugging Face
huggingface_model = pipeline('token-classification', model='markusiko/rubert-base-punctuation')

def create_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user TEXT,
            input_message TEXT,
            result TEXT,
            model_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_db()

def log_to_db(user, input_message, result, model_type):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (user, input_message, result, model_type)
        VALUES (?, ?, ?, ?)
    ''', (user, input_message, result, model_type))
    conn.commit()
    conn.close()

def rubert_predict(text: str) -> str:
    '''
    Функция для исправления пунктуационных ошибок моделью RuBERT
    '''
    new_text = ''
    tokens_predicted = huggingface_model(text, aggregation_strategy='simple')
    current_idx = 0
    for i in range(len(tokens_predicted)):
        if i != len(tokens_predicted) - 1:
            if tokens_predicted[i + 1]['word'].startswith('##'):
                continue
        new_text += text[current_idx:tokens_predicted[i]['end']] + tokens_predicted[i]['entity_group']
        current_idx = tokens_predicted[i]['end']
    return new_text

# Создание клавиатуры для выбора модели
keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text='Использовать GigaChat')],
        [KeyboardButton(text='Использовать RuBert')]
    ],
    resize_keyboard=True
)

# Переменная для хранения выбранной модели
user_model_choice = {}

async def main():
    await dp.start_polling(bot)

async def gigachat_correction(text):
    logging.info(f"Получен текст для GigaChat: {text}")
    try:
        # Авторизация в сервисе GigaChat
        chat = GigaChat(model='GigaChat:latest',
                        credentials=GIGACHAT_TOKEN,
                        verify_ssl_certs=False)

        messages = [SystemMessage(content=
                                  '''Расставь в тексте знаки препинания,\
                                  чтобы они соответствовали нормам русского языка'''),
                    HumanMessage(content='Текст: \n' + text)]
        answer = chat.invoke(messages).content
        logging.info(f"Ответ от GigaChat: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Ошибка при обращении к GigaChat: {e}")
        return "Произошла ошибка при обработке вашего запроса. Попробуйте позже."

async def rubert_correction(text):
    logging.info(f"Получен текст для RuBert: {text}")
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, rubert_predict, text)
        logging.info(f"Ответ от RuBert: {result}")
        return result
    except Exception as e:
        logging.error(f"Ошибка при обращении к RuBert: {e}")
        return "Произошла ошибка при обработке вашего запроса. Попробуйте позже."

hello_text = 'Отправь мне текстовое сообщение и я попробую исправить в нем пунктуационные ошибки.\nВыбери модель: GigaChat или дообученный RuBert.'

async def send_welcome_message(message: types.Message):
    await message.answer(f'Привет, {message.from_user.full_name}!\n{hello_text}', reply_markup=keyboard)

@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    await send_welcome_message(message)

@dp.message(Command('help'))
async def cmd_help(message: types.Message):
    await send_welcome_message(message)

@dp.message()
async def correct_punctuation(message: types.Message):
    user_id = message.from_user.id
    if message.text == 'Использовать GigaChat':
        user_model_choice[user_id] = 'gigachat'
        await message.answer('Теперь я буду использовать GigaChat для исправления пунктуации.')
    elif message.text == 'Использовать RuBert':
        user_model_choice[user_id] = 'rubert'
        await message.answer('Теперь я буду использовать RuBert для исправления пунктуации.')
    else:
        model_choice = user_model_choice.get(user_id, 'gigachat')
        if model_choice == 'gigachat':
            corrected_text = await gigachat_correction(message.text)
        else:
            corrected_text = await rubert_correction(message.text)
        await message.answer(corrected_text)
        log_to_db(message.from_user.username, message.text, corrected_text, model_choice)

if __name__ == '__main__':
    asyncio.run(main())
