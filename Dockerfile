FROM python:3.9
WORKDIR /deploy
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 8080
RUN nomic login "api key"
ENTRYPOINT ["streamlit", "run", "--server.port", "8080"]
CMD ["deploy/app.py"]