from .BaiduAPI import BaiduAPI

class Llma_2(BaiduAPI):
    def __init__(self):
        super().__init__()
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/qianfan_chinese_llama_2_7b?access_token=" + self.access_token
    

