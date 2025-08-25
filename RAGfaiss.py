import datasets
import faiss
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
import re

data_path = "./dataset"

ds=datasets.load_from_disk(data_path)
ds=ds['train']
ds=ds.select(range(50000))
def split_text(text):
    sentences=re.split(r"(?<=[。！？])",text)
    return [s.strip() for s in sentences if s.strip()]

def process(example):
    text= example['summary'] if example['summary'] else ""
    if example['sections'] and example['sections'][0]['title'] and example['sections'][0]['title']=='简介':
        text += example['sections'][0]['content']
    if not text:
        return
    return {
        'content':f"{example['title']}:{text}"
        }

mapped_ds=ds.map(process,remove_columns=ds.column_names)

embed_model = SentenceTransformer("BAAI/bge-large-zh")
embedding_content=embed_model.encode(ds,batch_size=64,convert_to_numpy=True)

dim = embedding_content.shape[1]
index = faiss.IndexFlatL2(dim)   # 简单L2索引
index.add(embedding_content)

# 建立 id → 文本 映射
id_map = {i: {"id": str(uuid.uuid4()), "text": ds[i]} for i in range(len(ds))}

faiss.write_index(index, "docs.faiss")
np.save("id_map.npy", id_map, allow_pickle=True)