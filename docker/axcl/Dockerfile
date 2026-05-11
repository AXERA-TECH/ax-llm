FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Context layout (staged by CI or user):
# - ./axllm
# - ./axcl/lib/*.so
COPY axllm /usr/local/bin/axllm
COPY axcl/lib/ /opt/axcl/lib/

ENV LD_LIBRARY_PATH=/opt/axcl/lib

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/axllm"]
