from celery import Celery
from app import utils
'''
from app.utils import process_file, fine_tune_model
'''
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

celery = Celery('worker', 
                broker='redis://redis:6379/0', 
                backend='redis://redis:6379/0')

@celery.task
def generate_training_data_task(directory: str, job_id: str):
    model_name = "google/gemma-2b"  # Update with your model path or name
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    output_dir = f"/data/{job_id}/train"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        utils.process_file(file_path, model, tokenizer, output_dir)
    return "Training data generation completed."

@celery.task
def fine_tune_model_task(training_data_path: str, model_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    utils.fine_tune_model(training_data_path, model_name, output_dir)
    return f"Fine-tuning completed. Model saved to {output_dir}."

