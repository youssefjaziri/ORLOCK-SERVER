import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orlock.brain.messages_in_brain import MsgInBrain


def test_msg_in_brain_update():
    msg = MsgInBrain(
        date="2024-06-01",
        hour="10:00",
        user="Alice",
        user_msg="Hello"
    )

    msg.update("user_msg", "Updated message")

    value = msg.get_value()

    assert value["user_msg"] == "Updated message"
    assert msg.get_field("user_msg") == "Updated message"

    return msg


if __name__ == "__main__":
    msg = test_msg_in_brain_update()
    msg.print_msg()
    msg.update("intention", "Greetting")
    print("\nAfter update:\n")
    msg.print_msg()
    print("Teste passou com sucesso.")