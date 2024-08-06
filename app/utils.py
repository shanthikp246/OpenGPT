import os
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

HF_TOKEN = os.getenv("HF_TOKEN")

def chunk_file(file_path, chunk_size=512):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

def generate_questions(chunk, model, tokenizer):
    input_text = f"Generate 3 questions based on the following text:\n{chunk.strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, num_return_sequences=3)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def generate_answers(chunk, questions, model, tokenizer):
    qa_pairs = []
    for question in questions:
        input_text = f"Question: {question}\nAnswer using the information provided in the following text:\n{chunk.strip()}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

def process_file(file_path, model, tokenizer, output_dir):
    for chunk in chunk_file(file_path):
        questions = generate_questions(chunk, model, tokenizer)
        qa_pairs = generate_answers(chunk, questions, model, tokenizer)
        output_file = os.path.join(output_dir, os.path.basename(file_path) + ".qa")
        with open(output_file, 'a', encoding='utf-8') as out_file:
            for pair in qa_pairs:
                out_file.write(json.dumps(pair) + "\n")

def load_model(job_id):
    model_dir = f"/data/{job_id}/model"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def answer_question(model, tokenizer, question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def fine_tune_model(training_data_path, model_name, output_dir):
    dataset = load_dataset('json', data_files={'train': training_data_path})
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for param in model.parameters():
        param.requires_grad = False  # Freeze the model weights
    
    # Configure and add LoRA adapter
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train']
    )
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save additional metadata
    metadata = {
        "model_name": model_name,
        "adapter_type": "LoRA",
        "peft_config": peft_config.__dict__
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

