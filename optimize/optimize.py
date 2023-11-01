from optimize.prompt import OptimizePrompt
from optimize.vote import Vote
from modelAPI.Baidu.Llma_2 import Llma_2
from modelAPI.Baidu.ERNIE_Bot import ERNIE_Bot
from modelAPI.Baidu.ChatGLM import ChatGLM
from modelAPI.OpenAI.Chatgpt import Chatgpt
from concurrent.futures import ThreadPoolExecutor, as_completed

TEST = False

class Optimize():
    def __init__(self) -> None:

        self.optPrompt = OptimizePrompt()
        self.vote = Vote()
        self.user_input = "长沙有几所985大学呢,我希望你能详细介绍一下？"
        # self.user_input = "太阳是什么类型？"
    
    def bot_optimize_prompt(self, prompt=None):

        if prompt is None:
            prompt = self.get_prompts()
        
        bot = ERNIE_Bot()
        response = bot.response(prompt)
        
        # 获取“优化结果：”后的字符串切片
        optimized_result = response[response.index("优化结果") + len("优化结果")+1:].strip()
        # print(optimized_result)
        
        end_index = optimized_result.find("\n")
        if end_index != -1:
            optimized_result = optimized_result[:end_index]
        
        end_index = optimized_result.find("。")
        if end_index != -1:
            optimized_result = optimized_result[:end_index]

        return optimized_result

    def get_prompts(self, user_input=None, label=None):
        
        if user_input is None: 
            user_input = self.user_input

        prompts = [user_input]

        opt_prompt = self.optPrompt.optimize_prompt(user_input,label)
        prompts.append(self.bot_optimize_prompt(opt_prompt))

        #重复一次用户输入，检测稳定性
        prompts.append(user_input)

        keywords = self.optPrompt.extract_keywords(user_input)
        prompts.append(" ".join(keywords))
        
        return prompts

    def get_responses(self, prompts=None):

        if prompts is None:
            prompts = self.get_prompts()

        responses_all = []
        weights = {}
        weights_all = []

        def ernie_bot():

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(ERNIE_Bot().response, prompt) for prompt in prompts]
                responses = [future.result() for future in as_completed(futures)]

            print("ERNIE_Bot:")
            for resp in responses:            
                print("__________________________________________________________")
                if resp != None:
                    print("\n"+resp)
            weight = self.vote.stability(responses)
            weights["ERNIE_Bot"] = weight
            if TEST:
                print("ERNIE_Bot稳定性为：",weight)
            weights_all.extend([weight]*len(responses))
            return responses
        
        def chatglm():
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(ChatGLM().response, prompt) for prompt in prompts]
                responses = [future.result() for future in as_completed(futures)]

            print("chatglm:")
            for resp in responses:            
                print("__________________________________________________________")
                if resp != None:
                    print("\n"+resp)
            weight = self.vote.stability(responses)
            weights["ChatGLM"] = weight
            if TEST:
                print("ChatGLM",weights)
            weights_all.extend([weight]*len(responses))
            return responses
        
        def llma_2():
            # with ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(Llma_2().response, prompt) for prompt in prompts]
            #     responses = [future.result() for future in as_completed(futures)]
            responses = []
            for prompt in prompts:
                responses.append(Llma_2().response(prompt))
            print("llma_2:")
            for resp in responses:            
                print("__________________________________________________________")
                if resp != None:
                    print("\n"+resp)
            weight = self.vote.stability(responses)
            weights["Llma_2"] = weight
            if TEST:
                print("Llma_2",weight)
            weights_all.extend([weight]*len(responses))
            return responses
        
        def chatgpt():
            # with ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(Chatgpt().response, prompt) for prompt in prompts]
            #     responses = [future.result() for future in as_completed(futures)]
            responses = []
            for prompt in prompts:
                b = Chatgpt()
                a = b.response(prompt)
                responses.append(a)
            weight = self.vote.stability(responses)
            weights["ChatGPT"] = weight
            print("ChatGPT:")
            for resp in responses:            
                print("__________________________________________________________")
                if resp != None:
                    print("\n"+resp)
            if TEST:
                print("Chatgpt",weight)
            weights_all.extend([weight]*len(responses))
            return responses
            
        # with ThreadPoolExecutor() as executor:
            
        #     if TEST:
        #         futures = [executor.submit(bot) for bot in [ernie_bot]]

        #     else:
        #         futures = [executor.submit(bot) for bot in [ernie_bot, chatglm,llma_2,chatgpt]]

        #     for future in as_completed(futures):
        #         responses_all.extend(future.result())
        for bot in [ernie_bot, chatglm,llma_2,chatgpt]:
            responses_all.extend(bot())

        return responses_all, weights, weights_all
    
    def vote_responses(self):
        
        # 获取所有模型的回复
        responses_all, weights, weights_all = self.get_responses()

        #去除answers中的空
        answers = [answer for answer in responses_all if answer != ""]
        weights_all = [weight for weight in weights_all if weight != 0]
        print(weights)

        # 对回复进行投票
        vote_index = self.vote.majority_vote(answers,weights_all)

        return responses_all[vote_index]