FROM ghcr.io/josancamon19/posttrainbench-gpu:latest

COPY . /app/
RUN chmod -R a+rw /app/ && chmod +x /app/timer.sh
