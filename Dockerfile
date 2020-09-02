FROM python:3.7.9

COPY . /ocr

WORKDIR /ocr

RUN pip install -r requirements.txt
RUN pip install torch==0.4.1.post2

ENTRYPOINT ["python"]

CMD ["app.py"]


