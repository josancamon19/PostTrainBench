ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# --- RunPod SSH setup ---
# Enables SSH on RunPod pods. RunPod sets PUBLIC_KEY at runtime.
# When PUBLIC_KEY is unset (Docker/Modal), SSH is skipped and the
# container just sleeps (safe for docker-exec based environments).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh

RUN printf '#!/bin/bash\nset -e\n\
env >> /etc/environment\n\
if [ -n "$PUBLIC_KEY" ]; then\n\
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys\n\
  chmod 600 /root/.ssh/authorized_keys\n\
  [ -f /etc/ssh/ssh_host_ed25519_key ] || ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ""\n\
  [ -f /etc/ssh/ssh_host_rsa_key ] || ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ""\n\
  service ssh start\n\
fi\n\
exec sleep infinity\n' > /harbor-start.sh && chmod +x /harbor-start.sh

CMD ["/harbor-start.sh"]
