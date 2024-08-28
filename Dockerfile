FROM python:3.10.12
RUN pip install notebook==7.0.6
RUN pip install pandas==2.1.3
RUN pip install gensim==4.3.2
RUN pip install scipy==1.10.1
RUN pip install spacy==3.7.4
RUN pip install ipdb==0.13.13 
WORKDIR /veld/code/

