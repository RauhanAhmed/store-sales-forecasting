FROM python:3.11.5-slim
WORKDIR /api
COPY . /api
RUN apt-get update && apt-get install -y
RUN apt-get -y install curl
RUN apt-get install libgomp1
RUN pip3 install pandas fastapi uvicorn u8darts lightgbm joblib libgomp1
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]