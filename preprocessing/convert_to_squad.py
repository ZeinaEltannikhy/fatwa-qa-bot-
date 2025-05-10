import json
import uuid
import logging
from fuzzywuzzy import fuzz
from camel_tools.utils.normalize import normalize_alef_ar, normalize_teh_marbuta_ar

# Set up logging
logging.basicConfig(level=logging.INFO)

def normalize_arabic(text):
    """Normalize Arabic text for consistent matching."""
    text = normalize_alef_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text.strip()

def find_answer_span(context, answer, threshold=80):
    """
    Find the answer span in the context using fuzzy matching.
    Returns (answer_text, start_index) or (None, None) if not found.
    """
    context = normalize_arabic(context)
    answer = normalize_arabic(answer)
    
    # Use the entire chunk as a fallback answer
    max_score = fuzz.partial_ratio(answer, context)
    best_match = answer
    start_index = 0
    
    # Try to find a shorter span with high similarity
    context_words = context.split()
    for i in range(len(context_words)):
        for j in range(i + 1, len(context_words) + 1):
            candidate = " ".join(context_words[i:j])
            score = fuzz.partial_ratio(answer, candidate)
            if score > max_score and score >= threshold:
                max_score = score
                best_match = candidate
                start_index = context.find(candidate)
    
    if max_score >= threshold:
        return best_match, start_index
    else:
        logging.warning(f"Could not find answer span for context: {context[:50]}...")
        return context, 0  # Fallback to full context

def convert_to_squad(input_file, output_file):
    """Convert fatwas JSONL to SQuAD 2.0 format."""
    squad_data = {"version": "v2.0", "data": []}
    title_to_paragraphs = {}
    skipped = 0
    
    # Read input JSONL
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                required_fields = ["title", "question", "chunk"]
                missing = [field for field in required_fields if field not in entry or not entry[field]]
                
                if missing:
                    logging.warning(f"Skipping line {i}: Missing fields {missing}")
                    skipped += 1
                    continue
                
                if len(entry["chunk"].strip()) < 20 or entry["chunk"].strip() == "المحتويات":
                    logging.warning(f"Skipping line {i}: Invalid context: {entry['chunk'][:50]}...")
                    skipped += 1
                    continue
                
                title = entry["title"]
                question = entry["question"]
                context = entry["chunk"]
                
                # Initialize title in squad_data if not present
                if title not in title_to_paragraphs:
                    title_to_paragraphs[title] = {
                        "title": title,
                        "paragraphs": []
                    }
                
                # Find or create paragraph for this context
                paragraph = None
                for p in title_to_paragraphs[title]["paragraphs"]:
                    if p["context"] == context:
                        paragraph = p
                        break
                if not paragraph:
                    paragraph = {"context": context, "qas": []}
                    title_to_paragraphs[title]["paragraphs"].append(paragraph)
                
                # Extract answer span
                answer_text, answer_start = find_answer_span(context, context)  # Use chunk as answer source
                qa_id = str(uuid.uuid4())
                
                qa = {
                    "question": question,
                    "id": qa_id,
                    "answers": [{
                        "text": answer_text,
                        "answer_start": answer_start
                    }],
                    "is_impossible": False
                }
                paragraph["qas"].append(qa)
                
            except json.JSONDecodeError:
                logging.warning(f"Skipping line {i}: Invalid JSON")
                skipped += 1
                continue
    
    # Convert title_to_paragraphs to squad_data
    squad_data["data"] = list(title_to_paragraphs.values())
    
    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Converted data saved to {output_file}")
    logging.info(f"Total titles: {len(squad_data['data'])}")
    logging.info(f"Total QAs: {sum(len(p['qas']) for t in squad_data['data'] for p in t['paragraphs'])}")
    logging.info(f"Total entries skipped: {skipped}")

if __name__ == "__main__":
    convert_to_squad("data/processed/fatwas_cleaned.jsonl", "data/processed/squad_fatwas_v2.json")