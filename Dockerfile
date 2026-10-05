# Image officielle de vLLM : même version que le worker Qwen de Teklia,
# qui fonctionne déjà sur l'agent pictoria-03.
FROM vllm/vllm-openai:v0.24.0

# L'image vLLM lance un serveur par défaut ; Arkindex appelle sa propre commande.
ENTRYPOINT []

ENV HF_HOME=/tmp/huggingface-cache \
    PYTHONUNBUFFERED=1

# Dépendance système de python-magic, importée par arkindex-base-worker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Arkindex exécute le conteneur avec UID 999 ; vLLM doit pouvoir le résoudre.
RUN useradd --uid 999 --no-user-group --home-dir /tmp/arkindex --no-create-home arkindex
ENV HOME=/tmp/arkindex \
    XDG_CACHE_HOME=/tmp/arkindex/.cache

WORKDIR /src
COPY pyproject.toml ./
COPY worker_pictoria_vlm ./worker_pictoria_vlm
RUN pip install --no-cache-dir .

# Vérifier les imports du worker pendant la construction.
RUN python3.12 -c "from worker_pictoria_vlm.worker import main"

# Donner au compte Arkindex les caches éventuellement créés pendant la construction.
RUN mkdir -p /tmp/arkindex/.cache && chown -R 999 /tmp/arkindex

CMD ["worker-pictoria-vlm"]
