from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import evaluate
import numpy as np

# 1. Load model and tokenizer
model_checkpoint = "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# 2. Load dataset
data_files = {"train": "data/processed/fatwas_squad_v2.json"}
raw_dataset = load_dataset("json", data_files=data_files, field="data")

# 3. Flatten nested SQuAD format
def extract_examples(batch):
    contexts, questions, answers = [], [], []
    for paragraphs in batch["paragraphs"]:  # this is a batch -> list of lists
        for paragraph in paragraphs:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if qa["is_impossible"]:
                    continue  # skip unanswerable
                for ans in qa["answers"]:
                    contexts.append(context)
                    questions.append(qa["question"])
                    answers.append(ans)
    return {"context": contexts, "question": questions, "answers": answers}

train_dataset = raw_dataset["train"].map(
    extract_examples,
    remove_columns=["title", "paragraphs"],
    batched=True
)

# 4. Tokenize and align
max_length = 384  # Max length of the input sequence
doc_stride = 128

def preprocess(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation=True,  # Automatically truncates both question and context
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=True
    )
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        # Find token start and end
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer out of span
        if not (start_char <= offsets[token_end_index][1] and end_char >= offsets[token_start_index][0]):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            for i in range(token_start_index, token_end_index + 1):
                if offsets[i][0] <= start_char < offsets[i][1]:
                    start_pos = i
                if offsets[i][0] < end_char <= offsets[i][1]:
                    end_pos = i
            start_positions.append(start_pos)
            end_positions.append(end_pos)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)

# 5. Training setup
training_args = TrainingArguments(
    output_dir="./araelectra-fatwas-qa",
    eval_strategy="no",  # Change `evaluation_strategy` to `eval_strategy`
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Try a smaller batch size
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=False,  # Disable mixed precision if you don't have a GPU
    report_to="none",
    gradient_accumulation_steps=3 # Accumulate gradients over 2 steps
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# 6. Start training
trainer.train()

# 7. Save model
trainer.save_model("araelectra-fatwas-qa")
