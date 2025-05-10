import os
import json
import logging
import requests  # To make HTTP requests to the FastAPI API
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Telegram bot token
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Safely get the token
TOKEN = os.getenv("BOT_TOKEN")


# FastAPI URL to call for the QA
API_URL = 'http://127.0.0.1:8000/answer'

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hello! Ask me any question and I will try to answer it based on fatwas.')

async def answer(update: Update, context: CallbackContext) -> None:
    question = update.message.text
    logger.info(f"Received question: {question}")

    try:
        response = requests.post(API_URL, json={"question": question})
        response.raise_for_status()
        answer_data = response.json()
        logger.info(f"API response: {answer_data}")

        answer = answer_data.get('answer', 'No answer found.')
        sources = '\n'.join(answer_data.get('source_urls', []))  # using 'source_urls' key
        reply_text = f"Answer: {answer}\n\nSources:\n{sources}"
        logger.info(f"Replying with: {reply_text}")
        await update.message.reply_text(reply_text)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        await update.message.reply_text("Sorry, something went wrong with the server.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await update.message.reply_text("An unexpected error occurred.")



def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer))

    application.run_polling()

if __name__ == '__main__':
    main()
