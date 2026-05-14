import os

class ProcessMsgInBrain:
    VALID_LABELS = {"Greeting", "WhereIs", "PathTo", "Wait", "EscortUser"}

    def __init__(self, msg_in_brain, llm_client, path=None):
        self.msg_in_brain = msg_in_brain
        self.llm_client = llm_client

        if path is None:
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, "prompts", "user_intention.txt")

        self.intent_prompt = self._load_prompt(path)

    def _load_prompt(self, prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Erro ao ler o ficheiro de prompt: {e}")
            return ""

    def _normalize_label(self, response):
        if not response:
            return None

        response = response.strip()

        if response in self.VALID_LABELS:
            return response

        for label in self.VALID_LABELS:
            if response.lower() == label.lower():
                return label

        for label in self.VALID_LABELS:
            if label.lower() in response.lower():
                return label

        return None

    def process_msg_llm(self):
        if not self.msg_in_brain.value:
            return None

        last_msg = self.msg_in_brain.value[-1]
        user_message = last_msg.get("user_msg")

        if not user_message:
            return None

        raw_response = self.llm_client.chat([
            {"role": "system", "content": self.intent_prompt},
            {"role": "user", "content": f"Frase do utilizador:\n{user_message}"}
        ])

        label = self._normalize_label(raw_response)

        last_msg["orlock_response"] = label
        return label