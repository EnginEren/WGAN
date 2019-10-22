FROM pytorch/pytorch:latest

#Install dependencies

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt 
RUN pip install --upgrade --no-cache-dir -r requirements.txt && \     
	rm requirements.txt

RUN yum -y update \
    && yum install -y \
                	git \
					vim 

WORKDIR /home

