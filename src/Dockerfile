# Base off of the Python image
FROM python:3.6.6-slim-stretch
LABEL maintainer "Pedro Teixeira <pedrotei@hutoma.com>"

RUN echo "2018-07-04" > /os_patch_date.txt

RUN apt-get update \
 && apt-get upgrade -y \
 && apt install -y --no-install-recommends \
  build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install pipenv
ENV PIP_NO_CACHE_DIR=false
RUN pip install --upgrade pip pipenv

# Apply the pipenv
RUN mkdir -p /src/embedding
COPY Pipfile* setup.py /src/
WORKDIR /src
RUN pipenv install --system

COPY initialize_model_data.py /src/
RUN python initialize_model_data.py

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

# Get the NLTK corpus data for appuser
RUN python -m nltk.downloader stopwords brown

ENV EMB_SERVER_PORT 9090

EXPOSE 9090

CMD ["python3", "/src/embedding/server.py"]