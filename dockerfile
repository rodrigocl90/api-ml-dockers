FROM python:3.12.11-slim  

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py /app/api.py
COPY model.pkl /app/artifacts/model.pkl
COPY model_card.json /app/artifacts/model_card.json

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "api:app"]
