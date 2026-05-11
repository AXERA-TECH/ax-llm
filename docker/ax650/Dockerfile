FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Context layout (staged by CI or user):
# - ./axllm
COPY axllm /usr/local/bin/axllm

# AX650 runtime libraries are typically provided by the board rootfs under /soc.
# Run container with: `-v /soc:/soc:ro -e LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH`.
ENV LD_LIBRARY_PATH=/soc/lib

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/axllm"]
