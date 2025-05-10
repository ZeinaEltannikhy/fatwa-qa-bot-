from qa_model.retriever import retrieve_documents  # Call the retriever from your retriever.py
from qa_model.generator import generate_answer  # Call the generator from your generator.py

def get_answer(question: str):
    """
    This is the RAG pipeline that uses both retriever and generator to answer a question.
    It first retrieves relevant documents and then generates an answer.
    """
    # Step 1: Retrieve the most relevant documents based on the question
    documents = retrieve_documents(question)

    # Step 2: Use the generator (QA model) to generate the final answer based on the retrieved documents
    answer = generate_answer(question, documents)
    
    return {
        "question": question,
        "answer": answer["answer"],
        "source_urls": answer["sources"]
    }
