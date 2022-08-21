FROM python:3.8-slim-buster

RUN apt-get update

RUN mkdir -p /app

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY ./src /app

CMD ["python", "/app/run_longformer_marco.py"]
