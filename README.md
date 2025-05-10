
# Fatwa QA Bot 🤖📚

This is a project I developed to build an Arabic Question Answering system based on Islamic Fatwas. The system uses a Retrieval-Augmented Generation (RAG) approach combining semantic search and a fine-tuned transformer-based QA model. It’s deployed with FastAPI and integrated with a Telegram Bot for user-friendly interaction.

---

## 🚀 Features

- Arabic question-answering powered by AraElectra model  
- Semantic document retrieval using Sentence Transformers  
- Web API built with FastAPI  
- Telegram bot interface  
- Custom preprocessing and conversion to SQuAD format  
- Fine-tuning capability for Arabic QA datasets  

---

## 📂 Project Structure

- `qa_model/app.py`: FastAPI backend for receiving user questions  
- `bots/telegram_bot.py`: Telegram bot script to forward questions and return answers  
- `qa_model/pipeline.py`: Combines retriever and generator for full RAG pipeline  
- `qa_model/retriever.py`: Embedding-based retriever using Sentence Transformers  
- `qa_model/generator.py`: QA model powered by Hugging Face Transformers  
- `preprocessing/`: Scripts for cleaning and formatting raw fatwa data  
- `models/`: Script to fine-tune the QA model  
- `data/`: Contains original and processed fatwas  
- `config/`: Configuration file including model paths and bot token  

---

## 🛠️ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
````

---

## 💡 Usage

### ▶️ Start the FastAPI Server

To launch the backend API locally, run:

```bash
python qa_model/app.py
```

This will start the FastAPI server at:
[http://localhost:8000/answer](http://localhost:8000/answer)

You can send POST requests with a JSON body like:

```json
{
  "question": "ما حكم كتابة الآيات على الكفن؟"
}
```

---

### 🤖 Run the Telegram Bot

To start the Telegram bot, run:

```bash
python qa_model/telegram_bot.py
```

Make sure your bot token is configured in one of the following ways:

* Set inside `config/config.yaml` under the `telegram:` section
* Or export it as an environment variable named `BOT_TOKEN`

---

## 📜 License

This project is licensed under the **MIT License**.

You are free to:

* ✅ Use
* ✅ Modify
* ✅ Share
* ✅ Commercialize

As long as you:

* Include the original license
* Give proper credit to the original author

See the [LICENSE](LICENSE) file for full details.

```

---

