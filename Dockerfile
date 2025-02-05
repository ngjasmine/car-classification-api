FROM python:3.9-slim

WORKDIR /car-classification-api

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]