FROM python:3.9
WORKDIR /deploy
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 8080
RUN nomic login nk-6Ik6gEX8tixXIUWIrpMPu1fX176mO9KjC9J_n5Cr0DQ
ENTRYPOINT ["streamlit", "run", "--server.port", "8080"]
CMD ["deploy/app.py"]