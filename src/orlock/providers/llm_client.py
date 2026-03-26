import os
import requests
from typing import Any, Dict, List, Optional

class LocalLLMClient:
    """
    Minimal Ollama HTTP client.
    Configure with env vars:
      OLLAMA_BASE_URL (default http://localhost:11434)
      OLLAMA_MODEL    (default llama3.1)
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "mistral:latest")
        self.session = requests.Session()

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        r = self.session.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        chat_url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        r = self.session.post(chat_url, json=payload, timeout=120)

        # Se /api/chat não existir, fazemos fallback para /api/generate
        if r.status_code == 404:
            prompt = self._messages_to_prompt(messages)
            return self.generate(prompt, system=None, temperature=temperature)

        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # Conversão simples e robusta
        out = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                out.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                out.append(f"[USER]\n{content}\n")
            else:
                out.append(f"[ASSISTANT]\n{content}\n")
        out.append("[ASSISTANT]\n")
        return "\n".join(out)