import asyncio
import logging
import re
import pandas as pd
import os
# from joblib import load
# from navec import Navec
from punctuators.models import PunctCapSegModelONNX
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from keep_alive import keep_alive

keep_alive()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# model = load('log_reg.joblib')
# le = load('le.joblib')
#
# path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
# navec = Navec.load(path)

xlm_roberta = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
)


async def main():
    logging.basicConfig(level=logging.DEBUG)
    await dp.start_polling(bot)


# def preprocess_text(line):
#     '''
#     Функция для подготовки ответа логистической регрессии
#     '''
#     # обработка сообщения
#     line = re.sub('– ', '', line)
#     line = re.sub('— ', '', line)
#     #     line = re.sub('\(', '', line)
#     #     line = re.sub('\)', '', line)
#     line = re.sub('"', '', line)
#     line = line.lower()
#     line = re.sub("[^\w\s]", '', line)
#     line = re.sub('\s+', ' ', line)
#     if len(line) == 0:
#         return 'Попробуй отправить еще и немного букв'
#
#     # разбиение на токены и преобразование в эмбеддинги
#     tokens = [token for token in line.split(' ') if token != '']
#     embeds = []
#
#     for i in tokens:
#         try:
#             embeds.append(navec[i])
#         except:
#             embeds.append(navec['<unk>'])
#
#     embed_df = pd.DataFrame(embeds, columns=[f'embed_{i}' for i in range(300)])
#
#     # предсказания модели
#     preds = le.inverse_transform(model.predict(embed_df))
#     answer = ''
#     flg_new_sent = 1
#
#     for i in range(len(preds)):
#         token_to_add = tokens[i]
#
#         if flg_new_sent:
#             token_to_add = token_to_add[0].upper() + token_to_add[1:]
#
#         if preds[i] != 'o':
#             token_to_add += preds[i]
#
#         answer += token_to_add
#
#         if preds[i] in ['?', '...', '.', '!']:
#             flg_new_sent = 1
#         else:
#             flg_new_sent = 0
#
#             # если в конце нет завершающего знака, то ставим его
#             if i == (len(preds) - 1):
#                 answer += '.'
#
#         answer += ' '
#
#     return answer.strip()


def roberta_correction(text):
    print(text)
    text = re.sub("[^\w\s]", '', text)
    text = text.lower()
    if len(text) == 0:
        return 'Попробуй отправить еще и немного букв'
    results = xlm_roberta.infer(texts=[text], apply_sbd=True)
    print(' '.join(results[0]))
    return ' '.join(results[0])


hello_text = 'Отправь мне текстовое сообщение и я попробую исправить в нем пунктуационные ошибки'


@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    await message.answer(f'Привет, {message.from_user.full_name}!\n{hello_text}')


@dp.message()
async def correct_punctuation(message: types.Message):
    # await message.answer(preprocess_text(message.text))
    await message.answer(roberta_correction(message.text))

if __name__ == '__main__':
    asyncio.run(main())
