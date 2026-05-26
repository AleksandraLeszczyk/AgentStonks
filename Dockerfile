FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY marketview/ marketview/
COPY main.py .

EXPOSE 7860

CMD ["python", "main.py"]
