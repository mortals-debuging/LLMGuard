from .BaiduAPI import BaiduAPI

class ChatGLM(BaiduAPI):
    def __init__(self):
        super().__init__()
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/chatglm2_6b_32k?access_token=" + self.access_token
    