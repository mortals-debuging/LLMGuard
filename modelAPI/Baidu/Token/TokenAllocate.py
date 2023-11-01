# 初始化
import os
import json

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class Tokens(metaclass=Singleton):
    def __init__(self):
        self.configs = []
        self.index = 0
        for filename in os.listdir('./modelAPI/Baidu/Token'):
            if filename.endswith('.json'):
                with open(os.path.join('./modelAPI/Baidu/Token', filename), 'r') as f:
                    config = json.load(f)
                    API_KEY = config['API_KEY']
                    SECRET_KEY = config['SECRET_KEY']
                    APP_ID = config['APP_ID']
                    self.configs.append({'API_KEY': API_KEY, 'SECRET_KEY': SECRET_KEY, 'APP_ID': APP_ID})

    def get_keys(self):
        length = len(self.configs)
        if length == 0:
            raise Exception('No available keys')
        if length == 1:
            return self.configs[0]
        else:
            self.index += 1
            return self.configs[self.index % length]

def GetToken():
    token = Tokens()
    return token.get_keys()