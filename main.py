from modelAPI.Baidu.Llma_2 import Llma_2
from modelAPI.Baidu.ERNIE_Bot import ERNIE_Bot
from modelAPI.Baidu.ChatGLM import ChatGLM
from modelAPI.OpenAI.Chatgpt import ChatgptAPI
from concurrent.futures import ThreadPoolExecutor, as_completed

llma_2 = Llma_2()
ernie_bot = ERNIE_Bot()
chatglm = ChatGLM()
chatgpt = ChatgptAPI()
bots = [chatglm, ernie_bot, llma_2, chatgpt]

while True:
    user_input = input(">>>")
    if user_input == "exit":
        break
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(bot.response, user_input) for bot in bots]
        responses = [future.result() for future in as_completed(futures)]

    for i, response in enumerate(responses):
        print(f"ROBOT{i+1} >>>：{response}")
        bots[i].add_message(response, "assistant")