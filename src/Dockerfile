# Base off of the Python image
FROM python:3.5.5-slim
LABEL maintainer "Pedro Teixeira <pedrotei@hutoma.com>"

RUN apt-get update \
 && apt install -y --no-install-recommends \
  build-essential tk\
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
# Copy common code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Run script to download models for languages we require
RUN python -m spacy download en_core_web_md

# Copy the code
COPY embedding/ /src/embedding/

#---------------------------
# switch to non root user
# define user/group IDs as ARG
ARG USERID=1000
ARG GROUPID=1000
RUN addgroup --system --gid $GROUPID appuser
RUN adduser --system --uid $USERID --gid $GROUPID appuser

RUN mkdir /ai && chown appuser:appuser /ai

USER appuser
WORKDIR /home/appuser

# Get the NLTK corpus data
RUN python -m nltk.downloader stopwords brown

ENV EMB_SERVER_PORT 9090


EXPOSE 9090

CMD ["python3", "/src/embedding/server.py"]