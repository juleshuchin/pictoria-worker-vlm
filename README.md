# Worker pictorIA : VLM générique (vLLM)

Worker Arkindex qui envoie l'image de chaque élément à un modèle vision-langage servi par vLLM et enregistre la réponse comme transcription.

- `arkindex/worker.yml` : description du worker et champs du formulaire de configuration ;
- `worker_pictoria_vlm/worker.py` : code du worker (bibliothèque `arkindex-base-worker`) ;
- `Dockerfile` : image fondée sur `vllm/vllm-openai:v0.24.0` ;
- `.github/workflows/docker.yml` : construction de l'image et publication sur `ghcr.io/juleshuchin/pictoria-worker-vlm` (GitHub) ;
- `.gitlab-ci.yml` : même chose pour le registre du GitLab Huma-Num.

Image à déclarer dans Arkindex : `ghcr.io/juleshuchin/pictoria-worker-vlm:main` (ou l'étiquette d'une version, par exemple `:0.1.0`).
