import jieba
import jieba.analyse
import jieba.posseg as pseg

class OptimizePrompt:
    def __init__(self):
        
        self.label = ["生活常识","志愿填报","学校信息","其他"]

        self.prompt = "你现在的任务是优化调整句子，不要有任何的分析。你首先在（"+str(self.label)+"）中选择分类。格式为：分类结果： 优化结果：。这个句子是：\n\""
        jieba.load_userdict("optimize/PromptText/specialword.txt")
    
    def extract_keywords(self, text,n=3):

        #jieba已经加载了词典
        keywords = jieba.analyse.extract_tags(text, topK=n)
        # print("/".join(keywords))
        return keywords

    def extract_keywords_with_pos(self, text, keywords):

        # 直接对整个文本进行词性标注
        words_with_pos = [(word, flag) for word, flag in pseg.cut(text) if word in keywords]

        return words_with_pos

    def optimize_prompt(self, user_input=None, label=None):

        if user_input is None:
            raise Exception("user_input is None")
        if label is None:
            label = self.label

        # 提取关键词
        keywords = self.extract_keywords(user_input)

        # 提取实体
        entities = self.extract_keywords_with_pos(user_input,keywords)
        # print(entities)
        
        # 生成优化后的prompt
        prompt = self.prompt + user_input +"\""

        for entity in entities:
            entity = entity[0]
            prompt = prompt.replace(entity, "["+entity+"]")
            
        return prompt
    
    