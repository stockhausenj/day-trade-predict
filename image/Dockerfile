FROM python:3.12

WORKDIR /opt/ml/code

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "train.py"]
