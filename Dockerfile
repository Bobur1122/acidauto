FROM mcr.microsoft.com/playwright/python:v1.64.1-noble

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 10000
CMD ["python", "app.py"]
