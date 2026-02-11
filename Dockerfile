FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create instance directory for database
RUN mkdir -p /app/instance && chmod 755 /app/instance

ENV PYTHONPATH=/app

EXPOSE 5000

CMD ["python", "-m", "src.app"]