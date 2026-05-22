from orlock.providers.llm_client import LocalLLMClient
from orlock.schemas.llm import MessageToLLMRequest

class LLMService:
    def __init__(self):
        self.client = LocalLLMClient()

    def message_to_llm(self, payload: MessageToLLMRequest) -> str:
        # If history exists, do chat; else do generate
        if payload.history:
            messages = []
            if payload.system:
                messages.append({"role": "system", "content": payload.system})
            messages.extend([m.model_dump() for m in payload.history])
            messages.append({"role": "user", "content": payload.prompt})

            return self.client.chat(messages, temperature=payload.temperature)

        return self.client.generate(payload.prompt, system=payload.system, temperature=payload.temperature)

    def message_to_llm_text(self, prompt: str, system: str = None, temperature: float = 0.2) -> str:
        """Simple text-to-text LLM call (used by orchestration layer)."""
        return self.client.generate(prompt, system=system, temperature=temperature)