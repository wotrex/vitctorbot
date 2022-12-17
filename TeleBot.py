import telebot
import pymongo
from PIL import Image
import io
from generation3 import generate
from time import time

bot = telebot.TeleBot('5633914858:AAGJSmcKFbFgdu1V62fRLJn9xBcjQNADx_Y')
db_client = pymongo.MongoClient("mongodb+srv://wotrex:i98wbdz9@victorgame.2kqtytt.mongodb.net/?retryWrites=true&w=majority")
dat = db_client['VictorGame']
last_message = 0

def sendMediaGroup(chatid):
    data = dat['maps'].find_one({'id': chatid})['terrain']
    data2 = dat['maps'].find_one({'id': chatid})['without_regions']
    data3 = dat['maps'].find_one({'id': chatid})['full_borders']
    data4 = dat['maps'].find_one({'id': chatid})['borders']
    print('send1')
    img = Image.open(io.BytesIO(data))
    img2 = Image.open(io.BytesIO(data2))
    img3 = Image.open(io.BytesIO(data3))
    img4 = Image.open(io.BytesIO(data4))
    media = [telebot.types.InputMediaPhoto(img), telebot.types.InputMediaPhoto(img2), telebot.types.InputMediaPhoto(img3), telebot.types.InputMediaPhoto(img4)]
    bot.send_media_group(chatid, media)

@bot.message_handler(commands=['generate'], chat_types = ['group','supergroup'])
def gen(message):
    global last_message
    if dat['Chats'].find_one({'id': message.chat.id}) != None:
        if dat['Chats'].find_one({'id': message.chat.id})['map'] == 'generating':
            last_message = bot.send_message(message.chat.id , "Мапа ще генерується")
        else:
            last_message = bot.send_message(message.chat.id , "Створюємо нову мапу(3-10 хвилин)")
            dat['Chats'].update_one({'id': message.chat.id},{ "$set": { 'time_to_gen': time() } })
            dat['Chats'].update_one({'id': message.chat.id},{ "$set": { 'map': 'generating' } })
            dat['maps'].delete_one({'id': message.chat.id})
            generate(message.chat.id)
            last_message = bot.send_message(message.chat.id , "Мапу створено")
            dat['Chats'].update_one({'id': message.chat.id},{ "$set": { 'map': 'ready' } })
            sendMediaGroup(message.chat.id)
        if time() - dat['Chats'].find_one({'id': message.chat.id})['time_to_gen'] > 2000:    ## Если в течении полу часа не сгенериться то можно еще сгенерить
            last_message = bot.send_message(message.chat.id , "Створюємо нову мапу(3-10 хвилин)")
            dat['Chats'].update_one({'id': message.chat.id},{ "$set": { 'time_to_gen': time() } })
            dat['Chats'].update_one({'id': message.chat.id},{ "$set": { 'map': 'generating' } })
            dat['maps'].delete_one({'id': message.chat.id})
            generate(message.chat.id)
            last_message = bot.send_message(message.chat.id , "Мапу створено")
            dat['Chats'].update_one({'id': message.chat.id},{ "$set": { 'map': 'ready' } })
            sendMediaGroup(message.chat.id)
    else:
        field = {
            'id':message.chat.id,
            'members':[message.from_user.id],
            'map': 'generating',
            'time_to_gen': time()
        }
        dat['Chats'].insert_one(field)
        last_message = bot.send_message(message.chat.id , "Створюємо мапу(3-10 хвилин)")
        generate(message.chat.id)
        last_message = bot.send_message(message.chat.id , "Мапу створено")
        dat['Chats'].update_one({'id': message.chat.id},{ "$set": { 'map': 'ready' } })
        sendMediaGroup(message.chat.id)


@bot.message_handler(commands=['show_map'], chat_types = ['group','supergroup'])
def show_map(message):
    sendMediaGroup(message.chat.id)

@bot.message_handler(content_types=['text'], chat_types = ['group','supergroup'])
def start(message):
    global last_message
    if message.text == '1':
        markup = telebot.types.ReplyKeyboardRemove()
        last_message = bot.send_message(message.chat.id , "ok", reply_markup=markup)
        bot.delete_message(message.chat.id, last_message.message_id)

bot.polling(none_stop=True, interval=0)