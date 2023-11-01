from .BaiduAPI import BaiduAPI

class ERNIE_Bot(BaiduAPI):
    def __init__(self):
        super().__init__()
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + self.access_token
    

