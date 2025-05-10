from qa_model.retriever import retrieve_documents
from transformers import pipeline

# Load Arabic QA model from Hugging Face directly (no need for local fine-tuned path)
qa_pipeline = pipeline(
    "question-answering",
    model="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA",
    tokenizer="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"
)

def generate_answer(question, documents, threshold=0.2):  # Lowered threshold for better recall
    """
    Generate an answer for the given question based on retrieved documents.

    Parameters:
    - question (str): The question to answer.
    - documents (list): A list of documents (each with 'text', 'title', 'url').
    - threshold (float): Confidence score threshold for considering an answer.

    Returns:
    - dict: Contains 'answer' and 'sources'.
    """
    answers = []
    for doc in documents:
        result = qa_pipeline(question=question, context=doc["text"])
        
        # Debug log
        print(f"\n📄 Title: {doc['title']}")
        print(f"📌 Score: {result['score']:.3f}, Answer: {result['answer']}")

        if result["answer"].strip() != "" and result["score"] > threshold:
            answers.append({
                "answer": result["answer"],
                "source": doc["url"],
                "score": result["score"]
            })

    if answers:
        # Return the top answer and its sources
        return {
            "answer": answers[0]["answer"],
            "sources": [ans["source"] for ans in answers]
        }
    else:
        # Return fallback if no good answers found
        return {
            "answer": "لم أجد إجابة واضحة على سؤالك في الوقت الحالي.",
            "sources": [doc["url"] for doc in documents]
        }
