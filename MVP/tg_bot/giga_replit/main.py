import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from keep_alive import keep_alive

keep_alive()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
GIGACHAT_TOKEN = os.environ.get('GIGACHAT_TOKEN')
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


async def main():
    await dp.start_polling(bot)


def gigachat_correction(text):
    print(text)
    # Авторизация в сервисе GigaChat
    chat = GigaChat(model='GigaChat:latest',
                    credentials=GIGACHAT_TOKEN,
                    verify_ssl_certs=False)

    messages = [SystemMessage(content="Расставь в тексте знаки препинания."),
                HumanMessage(content='Текст: \n' + text)]
    answer = chat.invoke(messages).content
    print(answer)

    return answer


hello_text = 'Отправь мне текстовое сообщение и я попробую исправить в нем пунктуационные ошибки'


@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    await message.answer(f'Привет, {message.from_user.full_name}!\n{hello_text}')


@dp.message()
async def correct_punctuation(message: types.Message):
    await message.answer(gigachat_correction(message.text))

if __name__ == '__main__':
    asyncio.run(main())
