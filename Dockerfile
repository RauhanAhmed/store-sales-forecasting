FROM python:3.11.5-slim
WORKDIR /api
COPY . /api
RUN pip3 install pandas joblib fastapi uvicorn u8darts lightgbm
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]