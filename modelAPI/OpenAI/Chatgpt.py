import openai
from collections import deque

from modelAPI.OpenAI.Token.TokenAllocate import GetToken


class ChatgptAPI:
    def __init__(self) -> None:
        token = GetToken()
        self.API_KEY = token['API_KEY']

        self.maxLenMessage = 50
        self.messageQueue = deque(maxlen=self.maxLenMessage)
        openai.api_key = self.API_KEY

    def add_message(self, mesg: str, role="user"):
        self.messageQueue.append({"role": role, "content": mesg})
        return list(self.messageQueue)

    def invoke(self, mesg: str, role="user"):
        self.add_message(mesg,role)
        messages= list(self.messageQueue)
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

    def response(self, mesg:str, role="user"):
        result =  self.invoke(mesg,role)
        print(result)
        return result.choices[0].message.content
