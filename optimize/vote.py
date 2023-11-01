import torch
import math
import statistics
from sentence_transformers import SentenceTransformer, util

class Vote:

    def cosine_similarity(self,answers):
        # 初始化Sentence-BERT模型
        model =  SentenceTransformer('distilbert-base-nli-mean-tokens')
        
        # 对回答进行编码
        embeddings = model.encode(answers)

        # 计算所有输出之间的余弦相似度
        cosine_sims = util.pytorch_cos_sim(embeddings, embeddings)
        return cosine_sims

    def stability(self,answers):

        cosine_sims  = self.cosine_similarity(answers)
        print("cosine_sims:",cosine_sims)
        # 将矩阵的下三角部分置为0
        cosine_sims = torch.triu(cosine_sims, diagonal=1)
        # 找到非0元素的索引
        non_zero_indices = torch.nonzero(cosine_sims)

        # 使用索引从矩阵中提取非0元素
        non_zero_values = cosine_sims[non_zero_indices[:, 0], non_zero_indices[:, 1]]

        # 计算均值和方差
        mean = torch.mean(non_zero_values).item()
        std = torch.std(non_zero_values).item()

        # 计算稳定性
        stability = math.exp(-std / mean)
        # print("稳定性为：", stability)
        return stability
    
    def majority_vote(self,answers,weights=None):
        print("answers",len(answers))
        if weights is None:
            weights = [1] * len(answers)

        # 将权重转换为张量
        weights_tensor = torch.tensor(weights)
        
        # 计算weights的转置
        weights_transpose_tensor = weights_tensor.unsqueeze(1).transpose(0, 1)

        cosine_sims  = self.cosine_similarity(answers)
        print("ALL cosine_sims:",cosine_sims)
        # 将矩阵中等于1的元素置为0
        cosine_sims = torch.where(cosine_sims == 1, torch.zeros_like(cosine_sims), cosine_sims)

        # 将矩阵中的元素与权重相乘
        cosine_sims = cosine_sims * weights_tensor * weights_transpose_tensor
        
        # 扁平化矩阵
        flattened = cosine_sims.flatten()

        # 对元素进行排序
        sorted_tensor, _ = torch.sort(flattened)
        # 删除所有1的元素
        sorted_tensor = sorted_tensor[sorted_tensor != 1]

        # 计算四分位数
        quartiles = statistics.quantiles(sorted_tensor, n=4)
        # 取四分位数的前三分之一部分
        three_quarters = quartiles[2]
        print("three_quarters",three_quarters)
        # 将矩阵中小于四分位数的元素置为0
        cosine_sims = torch.where(cosine_sims < three_quarters, torch.zeros_like(cosine_sims), cosine_sims)
        print("AFTER_cosine_sims",cosine_sims)
        #计算每一行的和
        row_sums = torch.sum(cosine_sims, dim=1)
        # 找到最大值的坐标
        max_indices = torch.where(row_sums == torch.max(row_sums))
        print("max_indices",max_indices)
        return max_indices[0].item()
    