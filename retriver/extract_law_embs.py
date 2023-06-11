from text2vec import SentenceModel
from text2vec import Word2Vec
import json
import numpy as np
import pickle
from tqdm import tqdm
import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

BS = 32
def compute_emb(law_data, model):
    global BS
    all_embs = None
    iteration_nums = len(law_data) // BS + 1
    # Embed a list of sentences
    for i in tqdm(range(iteration_nums)):
        batch_law_data = law_data[i*BS:i*BS+BS]
        sentences = []
        for j in range(len(batch_law_data)):
            sentences.append(batch_law_data[j][-1])
        sentence_embeddings = model.encode(sentences)
        if all_embs is not None:
            all_embs = np.concatenate([all_embs, sentence_embeddings], axis=0)
        else:
            all_embs = sentence_embeddings
    all_embs = torch.nn.functional.normalize(torch.tensor(all_embs)).numpy()
    return all_embs


if __name__ == "__main__":
    # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
    # device = torch.device("cpu")
    t2v_model = SentenceModel("shibing624/text2vec-base-chinese")
    law_data = json.load(open("./data/最核心法条_9k.json"))
    all_embs = compute_emb(law_data, t2v_model)
    pickle.dump(all_embs, open("./data/law_embs.pkl", 'wb'))
