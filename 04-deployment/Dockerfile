FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py app.py
COPY score.py score.py

RUN mkdir -p /app/outputs

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]