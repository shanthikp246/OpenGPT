from celery import Celery
import os
import time
import logging
from prometheus_client import CollectorRegistry, Counter, push_to_gateway
from celery.signals import celeryd_after_setup
import numpy as np
from model import Matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load broker and backend URLs from environment variables
broker_url = os.environ.get('CELERY_BROKER_URL', 'amqp://guest:guest@rabbitmq-service:5672//')
backend_url = os.environ.get('CELERY_RESULT_BACKEND', 'rpc://')

celery_app = Celery('worker', broker=broker_url, backend=backend_url)

registry = CollectorRegistry()
c = Counter('do_work_count', 'do work count', registry=registry)

# Connect signal to capture worker name
@celeryd_after_setup.connect
def capture_worker_name(sender, instance, **kwargs):
    os.environ["WORKER_NAME"] = '{0}'.format(sender)


@celery_app.task
def do_work(matrix1 , matrix2):
    try:
        nrowsA = len(matrix1)
        nrowsB = len(matrix2)
        job_name = os.environ["WORKER_NAME"]
        logging.info(f'Starting task with payload: {nrowsA=}, {nrowsB=}')
        A = np.array(matrix1)
        B = np.array(matrix2)
        C = np.dot(A, B)

        c.inc()
        
        push_to_gateway("http://pushgateway-service:9091", job=job_name, registry=registry)
        logging.info(f'Task completed with payload: {job_name}')
        return {"result": C.tolist()}
    
    except Exception as e:
        logging.error(f'Error occurred during task execution: {e}')
        return {"error": str(e)}
