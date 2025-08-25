# QLoRA-RAG
## 模型及数据集下载
微调模型和RAG基模型可使用model_download.py进行下载，任意文本生成模型理论上均可
另外需要下载BAAI/bge-large-zh用于RAG向量编码用于检索相似度
数据集可用dataset_download.py下载
## QLoRA
目前由于显存限制，QLoRA参数设置为r=4，α=16，target_modules=["q_proj","v_proj"]，若需要更强的性能可增加r和target_modules
同时由于时间限制，只选取了前500条数据进行了3轮训练
## RAG
向量和知识库分开存储按照id对应，也可直接改用Langchain封装的FAISS，分开存储的目前是为了通过title和summary高效检索并将完整信息传入prompt
