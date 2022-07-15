
FROM python:3.8-slim
WORKDIR /dockerimage
COPY ./app /dockerimage
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["python3", "app.py"]

#FROM python:3.8-slim

#COPY requirements.txt .

#RUN pip install -r requirements.txt 

#RUN mkdir -p app 

#COPY ./app app 

#EXPOSE 80 

#CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]
