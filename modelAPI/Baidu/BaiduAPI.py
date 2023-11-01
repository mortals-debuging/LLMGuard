import requests
import json
from collections import deque
from .Token.TokenAllocate import GetToken


class BaiduAPI:
    def __init__(self) -> None:
        token = GetToken()
        self.API_KEY = token['API_KEY']
        self.SECRET_KEY = token['SECRET_KEY']
        self.APP_ID = token['APP_ID']
        self.access_token = self.get_access_token()
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.url = ''

        self.maxLenMessage = 50
        self.messageQueue = deque(maxlen=self.maxLenMessage)
        
    def add_message(self, mesg:str, role="user"):
        self.messageQueue.append({'role': role, 'content': mesg})
        return {"messages": list(self.messageQueue)}
    
    def invoke(self, mesg:str, role="user"):
        self.add_message(mesg,role)
        if len(self.messageQueue) == 0:
            return None
        if self.url == '':
            raise Exception('No url')
        
        message = {"messages": list(self.messageQueue)}
        # print(message)
        payload = json.dumps(message)
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        return response.json()
    
    def response(self, mesg:str, role="user"):
        resp_json = self.invoke(mesg, role)
        # print(resp_json)
        return resp_json['result']

    def get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.API_KEY,
            "client_secret": self.SECRET_KEY
        }
        return str(requests.post(url, params=params).json().get("access_token"))