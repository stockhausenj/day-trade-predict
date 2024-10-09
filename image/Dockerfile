FROM python:3.12

WORKDIR /opt/ml/code

COPY . .

RUN pip install -r requirements.txt

ENV SAGEMAKER_PROGRAM=train.py

ENTRYPOINT ["python", "train.py"]
