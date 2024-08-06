from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from celery.result import AsyncResult
from uuid import uuid4
from app import tasks, utils
"""
from app.tasks import generate_training_data_task, fine_tune_model_task
from app.utils import load_model, answer_question
"""

app = FastAPI()

class TrainRequest(BaseModel):
    location: str

class FineTuneRequest(BaseModel):
    uuid: str
    model_name: str

class QuestionRequest(BaseModel):
    uuid: str
    question: str

@app.post("/generate_training_data")
def generate_training_data(train_request: TrainRequest, background_tasks: BackgroundTasks):
    if not train_request.location.startswith("/data"):
        raise HTTPException(status_code=400, detail="Invalid location. Must be within the /data directory.")
    
    job_id = str(uuid4())
    task = tasks.generate_training_data_task.apply_async(args=[train_request.location, job_id])
    background_tasks.add_task(task.wait)
    return {"job_id": job_id, "task_id": task.id}

@app.post("/finetune")
def finetune_model(fine_tune_request: FineTuneRequest, background_tasks: BackgroundTasks):
    training_data_path = f"/data/{fine_tune_request.uuid}/train"
    model_output_path = f"/data/{fine_tune_request.uuid}/model"
    if not os.path.exists(training_data_path):
        raise HTTPException(status_code=400, detail="Training data not found.")
    
    task = tasks.fine_tune_model_task.apply_async(args=[training_data_path, fine_tune_request.model_name, model_output_path])
    background_tasks.add_task(task.wait)
    return {"uuid": fine_tune_request.uuid, "task_id": task.id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        response = {"status": "Pending..."}
    elif task_result.state != 'FAILURE':
        response = {"status": task_result.state, "result": task_result.result}
    else:
        response = {"status": "Failed", "result": str(task_result.info)}
    return response

@app.post("/inference")
def ask_question(question_request: QuestionRequest):
    try:
        model, tokenizer = utils.load_model(question_request.uuid)
        answer = utils.answer_question(model, tokenizer, question_request.question)
        return {"question": question_request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

