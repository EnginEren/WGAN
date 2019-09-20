FROM pytorch/pytorch:latest

#Install dependencies

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt 
RUN pip install --upgrade --no-cache-dir -r requirements.txt && \     
	rm requirements.txt

RUN apt-get update 
RUN apt-get upgrade -y 
RUN apt-get install -y git
RUN apt-get install -y vim

WORKDIR /home

