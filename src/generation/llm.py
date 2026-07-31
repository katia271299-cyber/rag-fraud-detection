"""
src/generation/llm.py
Interface unifiée pour les LLMs : Ollama (local) et HuggingFace.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BaseLLM:
    """Interface commune pour tous les LLMs."""
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    """
    LLM via Ollama — exécution 100% locale.
    
    Prérequis :
        1. Installer Ollama : https://ollama.com
        2. Télécharger le modèle : ollama pull mistral
        3. Ollama doit tourner en arrière-plan
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.model       = model
        self.base_url    = base_url
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._verify_connection()

    def _verify_connection(self):
        """Vérifie qu'Ollama est accessible."""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model in m for m in models):
                logger.warning(f"Modèle '{self.model}' non trouvé. Lancez : ollama pull {self.model}")
            else:
                logger.info(f"✓ Ollama connecté — modèle '{self.model}' disponible")
        except Exception as e:
            logger.error(f"Ollama non accessible sur {self.base_url} : {e}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests

        # Construit un prompt texte brut compatible avec les modèles completion-only
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        full_prompt += f"{prompt}\n\nRéponse :"

        payload = {
            "model":  self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": ["Question :", "<|"],
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


class HuggingFaceLLM(BaseLLM):
    """
    LLM via HuggingFace Transformers — exécution locale ou cloud.
    Plus lent à initialiser mais ne nécessite pas Ollama.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cpu",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ):
        self.model_id       = model_id
        self.device         = device
        self.temperature    = temperature
        self.max_new_tokens = max_new_tokens
        self._pipeline      = None

    @property
    def pipeline(self):
        """Chargement paresseux du pipeline."""
        if self._pipeline is None:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Chargement du modèle HuggingFace : {self.model_id}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model     = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map=self.device,
            )
            logger.info("Modèle HuggingFace chargé.")
        return self._pipeline

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Formatage au format Mistral Instruct
        formatted = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        output = self.pipeline(
            formatted,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            return_full_text=False,
        )
        return output[0]["generated_text"].strip()


def create_llm(
    provider: str = "ollama",
    model: str = "mistral",
    temperature: float = 0.1,
    max_tokens: int = 512,
    **kwargs,
) -> BaseLLM:
    """Factory : crée le bon LLM selon la configuration."""
    if provider == "ollama":
        return OllamaLLM(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
    elif provider == "huggingface":
        return HuggingFaceLLM(model_id=model, temperature=temperature, max_new_tokens=max_tokens, **kwargs)
    else:
        raise ValueError(f"Provider inconnu : {provider}. Choisir 'ollama' ou 'huggingface'.")
