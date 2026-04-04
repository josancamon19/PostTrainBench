# Adds RunPod SSH layer on top of any GPU image (base or per-task)
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/ssh/ssh_host_*

ADD https://raw.githubusercontent.com/runpod/containers/main/container-template/start.sh /start.sh
RUN chmod +x /start.sh && \
    sed -i '/^start_nginx$/d' /start.sh

CMD ["/start.sh"]
