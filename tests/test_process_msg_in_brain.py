import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orlock.brain.process_msg_in_brain import ProcessMsgInBrain
from src.orlock.providers.llm_client import LocalLLMClient


class MsgInBrain:
    def __init__(self):
        self.value = []


def test_send_message_from_brain_to_llm():
    # memory is created and empty
    msg_in_brain = MsgInBrain()

    # whisper AI

    # test message is added to the brain
    msg_in_brain.value.append({
        "date": "2024-06-01",
        "hour": "10:00",
        "user": "Alice",
        "user_msg": "how are you?",
        "intention": None,
        "orlock_response": None
    })

    # create LLM client
    llm = LocalLLMClient()

    # Process the message in the brain and send to LLM
    processor = ProcessMsgInBrain(msg_in_brain, llm)

    # process message and get response from LLM
    response = processor.process_msg_llm()
    print(f"Resposta da LLM: {response}")

    # validate response
    assert response is not None
    assert isinstance(response, str)
    assert response.strip() != ""

    # verificar se a resposta foi guardada no brain
    saved_response = msg_in_brain.value[-1]["orlock_response"]

    assert saved_response is not None
    assert isinstance(saved_response, str)
    assert saved_response.strip() != ""


if __name__ == "__main__":
    test_send_message_from_brain_to_llm()
    print("Teste passou com sucesso.")