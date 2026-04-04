ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# --- RunPod SSH setup (matches official runpod/containers pattern) ---
# https://github.com/runpod/containers/blob/main/container-template/start.sh
# When PUBLIC_KEY is unset (Docker/Modal), SSH is skipped and the
# container just sleeps (safe for docker-exec based environments).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 -R /root/.ssh

COPY <<'HARBOR_START' /harbor-start.sh
#!/bin/bash
set -e

# --- SSH setup ---
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh

    [ -f /etc/ssh/ssh_host_rsa_key ]     || ssh-keygen -t rsa     -f /etc/ssh/ssh_host_rsa_key     -q -N ''
    [ -f /etc/ssh/ssh_host_dsa_key ]     || ssh-keygen -t dsa     -f /etc/ssh/ssh_host_dsa_key     -q -N ''
    [ -f /etc/ssh/ssh_host_ecdsa_key ]   || ssh-keygen -t ecdsa   -f /etc/ssh/ssh_host_ecdsa_key   -q -N ''
    [ -f /etc/ssh/ssh_host_ed25519_key ] || ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''

    service ssh start
fi

# --- Export env vars so SSH sessions inherit them ---
printenv | grep -E '^[A-Z_][A-Z0-9_]*=' | grep -v '^PUBLIC_KEY' | \
    awk -F = '{ val = $0; sub(/^[^=]*=/, "", val); print "export " $1 "=\"" val "\"" }' \
    > /etc/rp_environment
grep -q 'source /etc/rp_environment' ~/.bashrc 2>/dev/null || \
    echo 'source /etc/rp_environment' >> ~/.bashrc

# --- Pre/post start hooks ---
[ -f /pre_start.sh ] && bash /pre_start.sh
[ -f /post_start.sh ] && bash /post_start.sh

sleep infinity
HARBOR_START

RUN chmod +x /harbor-start.sh
CMD ["/harbor-start.sh"]
