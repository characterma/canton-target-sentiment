FROM python:3.8-slim-buster
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]

