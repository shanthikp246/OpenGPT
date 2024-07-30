FROM python:3.11.3-slim

WORKDIR /app

#install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

#copy project
COPY . .