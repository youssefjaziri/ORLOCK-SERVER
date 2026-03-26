

class MsgInBrain:
    def __init__(self, date, hour, user, user_msg):
        self.value = {
            "msg_id": f"{date}_{hour}_{user}",
            "date": date,
            "hour": hour,
            "user": user,
            "user_msg": user_msg,
            "intention": None,
            "orlock_response": None
        }

    def update(self, key, new_value):
        self.value[key] = new_value

    def get_value(self):
        return self.value

    def get_field(self, key):
        return self.value.get(key)

    def print_msg(self):
        for key, value in self.value.items():
            print(f"{key}: {value}")