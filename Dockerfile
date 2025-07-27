FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src     /app/src
COPY models  /app/models
COPY input   /app/input

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN useradd -m myuser \
 && chown -R myuser:myuser /app
USER myuser

ENTRYPOINT ["python3", "src/main.py", "/app/input"]
