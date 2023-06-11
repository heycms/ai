import faiss                   
import pickle
import argparse
import json
from text2vec import SentenceModel

def retriver(query):
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', default='/DB/rhome/yuhaowang/law_retriver/law_embs.pkl', type=str, help='')
    parser.add_argument('--rawdata_path', default='/DB/rhome/yuhaowang/law_retriver/最核心法条_9k.json', type=str, help='')
    parser.add_argument('--top_k', type=int, default=3, help='dst root to faiss database')
    args = parser.parse_args()

    law_embeds = pickle.load(open(args.embedding_path, 'rb'))
    raw_law_data = json.load(open(args.rawdata_path, 'rb'))
    
    index = faiss.IndexFlatIP(law_embeds.shape[-1])   
    # print(index.is_trained)
    index.add(law_embeds)                  
    # print(index.ntotal)

    t2v_model = SentenceModel("shibing624/text2vec-base-chinese")
    
    input_q = query
    while input_q != 'kill':
        q_emb = t2v_model.encode([input_q])
        D, I = index.search(q_emb, args.top_k)
        output = [raw_law_data[i] for i in I[0]]
        return output

    
