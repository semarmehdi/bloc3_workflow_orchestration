FROM python:3.11

WORKDIR /home/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nano unzip curl \
    && rm -rf /var/lib/apt/lists/*

COPY . . 

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install


COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV ARTIFACT_ROOT=$ARTIFACT_ROOT

CMD mlflow server -p $PORT \
    --host 0.0.0.0 \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --allowed-hosts "*"