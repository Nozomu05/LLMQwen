FROM python:3.11-slim

WORKDIR /app

COPY services/search/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ services/

CMD ["uvicorn", "services.search.app:app", "--host", "0.0.0.0", "--port", "8010"]
