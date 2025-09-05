FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# This includes `main.py`, the `src` folder, the `data` folder, and your `chroma_db`
COPY . .

EXPOSE 10000

# This runs your FastAPI server using Uvicorn on port 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]