# Image officielle de vLLM : même version que le worker Qwen de Teklia,
# qui fonctionne déjà sur l'agent pictoria-03.
FROM vllm/vllm-openai:v0.24.0

# L'image vLLM lance un serveur par défaut ; Arkindex appelle sa propre commande.
ENTRYPOINT []

ENV HF_HOME=/tmp/huggingface-cache \
    PYTHONUNBUFFERED=1

WORKDIR /src
COPY pyproject.toml ./
COPY worker_pictoria_vlm ./worker_pictoria_vlm
RUN pip install --no-cache-dir .

CMD ["worker-pictoria-vlm"]
