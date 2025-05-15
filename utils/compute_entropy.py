# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import json
import torch
import hdbscan
from utils.transformers_config import TransformersConfig
# from transformers_config import TransformersConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from scipy.stats import entropy
from collections import defaultdict


class ComputeEntropy:
    def __init__(self, model, tokenizer, top_k, n_clusters):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.n_clusters = n_clusters

    def calculate_entropy(self, probabilities):
        """计算概率分布的熵"""
        return entropy(probabilities, base=2)
        # return entropy(probabilities)

    def semantic_clustering(self, probs, embeddings):
        """对tokens的嵌入进行聚类"""
        clusterer = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        # clusterer = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=16)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
        labels = clusterer.fit_predict(embeddings)

        # 计算每个聚类token数目的概率分布
        # label_counts = np.bincount(labels, minlength=n_clusters)
        # cluster_probs = label_counts / len(labels)
        # print(cluster_probs)

        # 计算每个聚类token概率和的分布
        cluster_probs = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            cluster_probs[i] = probs[labels == i].sum().item()
        # 归一化
        cluster_probs /= cluster_probs.sum()
        return self.calculate_entropy(cluster_probs), labels

    def get_entropy(self, inputs):
        """生成文本并分析熵和语义聚类熵"""
        output = self.model(inputs)

        logits = output.logits[:, -1, :self.tokenizer.vocab_size][0]

        # 计算原始分布熵
        probabilities = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        token_entropy = self.calculate_entropy(probabilities)

        # 获取Top-k token及其概率
        top_k_indices = torch.topk(logits, self.top_k).indices.cpu().detach().numpy()
        top_k_probs = probabilities[top_k_indices]
        # top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices)

        # 获取Top-k token的嵌入
        token_embeddings = self.model.get_input_embeddings()(torch.tensor(top_k_indices).to(self.model.device))
        token_embeddings = token_embeddings.cpu().detach().numpy()

        # 进行语义聚类并计算聚类熵
        semantic_entropy, labels = self.semantic_clustering(top_k_probs, token_embeddings)

        return {
            "token_entropy": token_entropy,
            # "top_k_tokens": top_k_tokens,
            # "top_k_probs": top_k_probs,
            # 'labels': labels,
            "semantic_entropy": semantic_entropy,
        }


if __name__ == '__main__':
    # 加载模型和tokenizer
    # model_name = "/data/wangyidan/model/opt-6.7b"  # 替换为实际模型路径或名称
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained("/data/wangyidan/model/opt-6.7b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/data/wangyidan/model/opt-6.7b")

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=50272,
        device=device,
        max_new_tokens=200,
        min_length=230,
        do_sample=True,
        # no_repeat_ngram_size=4
    )

    # 示例调用
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # output_logits = model(**inputs).logits
    # last_logits = output_logits[0, -1, :]  # 获取最后一个token的logits

    print(tokenizer.vocab_size)

    compute_entropy = ComputeEntropy(model, tokenizer, top_k=tokenizer.vocab_size, n_clusters=300)
    result = compute_entropy.get_entropy(inputs['input_ids'])

    print("Token entropy:", result["token_entropy"])
    # print("Top-k tokens:", result["top_k_tokens"])
    # print("Labels:", result['labels'])
    print("Semantic entropy:", result["semantic_entropy"])
