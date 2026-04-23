# Adds RunPod SSH layer on top of any GPU image (base or per-task).
# Built once + pushed to GHCR; gpu-runpod tasks pin this image so harbor's
# _wait_for_ssh_ready doesn't hang 10+ min on raw runpod/pytorch images
# that don't ship with openssh-server or /harbor-start.sh.
ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/ssh/ssh_host_*

ADD https://raw.githubusercontent.com/runpod/containers/main/container-template/start.sh /start.sh
RUN chmod +x /start.sh && \
    sed -i '/^start_nginx$/d' /start.sh

CMD ["/start.sh"]
