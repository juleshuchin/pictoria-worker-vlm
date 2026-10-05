import base64
import io
import logging
import math
import os
import re

from arkindex_worker.models import Element
from arkindex_worker.worker import ElementsWorker

logger = logging.getLogger(__name__)

THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
TOKEN_KEYS = ("token", "hf_token", "HF_TOKEN", "hugging_face_token", "api_key")


def extract_token(secret) -> str | None:
    """Lit un jeton Hugging Face dans un secret Arkindex (texte, YAML ou JSON)."""
    if not secret:
        return None
    if isinstance(secret, str):
        return secret.strip() or None
    if isinstance(secret, dict):
        for key in TOKEN_KEYS:
            if secret.get(key):
                return str(secret[key]).strip()
        values = [v for v in secret.values() if isinstance(v, str) and v.strip()]
        if len(values) == 1:
            return values[0].strip()
    logger.warning("Le secret Hugging Face n'a pas un format reconnu ; il est ignoré.")
    return None


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class PictoriaVLMWorker(ElementsWorker):
    def configure(self):
        super().configure()

        token = extract_token(self.config.get("hf_token"))
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        # vLLM et transformers ne sont importés qu'ici, une fois l'environnement prêt
        from vllm import LLM, SamplingParams

        self.model_id = self.config["model"]
        self.system_prompt = (self.config.get("system_prompt") or "").strip()
        self.user_prompt = (self.config.get("user_prompt") or "").strip()
        assert self.user_prompt, "La consigne envoyée avec l'image ne peut pas être vide."
        # Compatibilité avec les consignes écrites pour le worker Qwen de Teklia
        self.user_prompt = self.user_prompt.replace("$IMAGE", "").strip()

        self.max_image_side = int(self.config.get("max_image_side") or 2000)
        self.thinking = bool(self.config.get("thinking", False))

        llm_kwargs = {
            "model": self.model_id,
            "max_model_len": int(self.config.get("max_model_len") or 16384),
            "gpu_memory_utilization": float(
                self.config.get("gpu_memory_utilization") or 0.92
            ),
            "enforce_eager": bool(self.config.get("enforce_eager", True)),
            "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
            "limit_mm_per_prompt": {"image": 1},
        }
        quantization = self.config.get("quantization") or "auto"
        if quantization != "auto":
            llm_kwargs["quantization"] = quantization

        logger.info(f"Chargement du modèle {self.model_id} avec {llm_kwargs}")
        self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            max_tokens=int(self.config.get("max_new_tokens") or 4096),
            temperature=float(self.config.get("temperature") or 0.0),
            logprobs=1,
        )
        logger.info("Modèle chargé.")

    def build_messages(self, image) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    {"type": "text", "text": self.user_prompt},
                ],
            }
        )
        return messages

    def load_image(self, element: Element):
        try:
            image = element.open_image(
                max_width=self.max_image_side, max_height=self.max_image_side
            )
        except NotImplementedError:
            # Le serveur IIIF impose un téléchargement par tuiles : on réduit ensuite
            image = element.open_image()
        image.thumbnail((self.max_image_side, self.max_image_side))
        return image

    def process_element(self, element: Element):
        image = self.load_image(element)
        logger.info(f"Image de {image.width} × {image.height} pixels envoyée au modèle")

        outputs = self.llm.chat(
            self.build_messages(image),
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": self.thinking},
            use_tqdm=False,
        )
        completion = outputs[0].outputs[0]
        text = THINK_BLOCK.sub("", completion.text).strip()

        if not text:
            logger.warning(f"Réponse vide pour l'élément {element.id} ; rien n'est enregistré.")
            return

        # Confiance : probabilité moyenne par token (moyenne géométrique)
        confidence = 1.0
        if completion.cumulative_logprob is not None and completion.token_ids:
            confidence = math.exp(
                completion.cumulative_logprob / len(completion.token_ids)
            )
        confidence = float(min(max(confidence, 0.0), 1.0))

        self.create_transcription(element, text=text, confidence=confidence)
        logger.info(
            f"Transcription de {len(text)} caractères créée (confiance {confidence:.2f})"
        )


def main():
    PictoriaVLMWorker(
        description="Worker pictorIA pour les modèles vision-langage servis par vLLM"
    ).run()


if __name__ == "__main__":
    main()
