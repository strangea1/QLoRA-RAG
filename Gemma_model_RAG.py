from transformers import AutoTokenizer
from llama_cpp import Llama
import uuid
from sentence_transformers import SentenceTransformer
import faiss
from langgraph.graph import StateGraph, END
from langchain.llms.base import LLM
from typing import TypedDict, List, Optional
import numpy as np
from langgraph.checkpoint.memory import MemorySaver

model_path = "./model/gemma3_4b/gemma-3-4b-it-q4_0.gguf"
embed_path = "./model/bge-large"
model = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8
)

class LlamaWrapper(LLM):
    model: object

    @property
    def _llm_type(self) -> str:
        return "llama_cpp"

    def _call(self, prompt: str,max_tokens=256,stop: Optional[List[str]] = None) -> str:
        token = self.model.tokenize(prompt.encode("utf-8"))
        generated_tokens = []
        # 手动限制生成长度
        for i, token in enumerate(model.generate(token, top_k=40, top_p=0.95, temp=0.7, repeat_penalty=1.1)):
            generated_tokens.append(token)
            if i >= max_tokens - 1:
                break
        text = self.model.detokenize(generated_tokens)
        return text.decode("utf-8")
    
llm=LlamaWrapper(model=model)

embed_model = SentenceTransformer("BAAI/bge-large-zh",cache_folder=embed_path)
index = faiss.read_index("docs.faiss")
id_map=np.load("id_map.npy",allow_pickle=True).item()

class MemoryState(TypedDict):
    n:int
    query:str
    chat_history:List[str]
    knowledge:List[str]
    answer:str

def retrieve_node(state):
    query=state['query']
    query_vec = embed_model.encode(query)
    query_vec = np.array([query_vec]).astype('float32')
    _,I=index.search(query_vec,k=5)
    ids = I[0].tolist() 
    return {'knowledge':[id_map[i] for i in ids]}

def llm_node(state: MemoryState):
    query = state["query"]
    chat_history = state["chat_history"]
    docs = state["knowledge"]
    for d in docs:
        context = "\n".join(d["text"])
    history = "\n".join(chat_history[-5:])  # 短期记忆，保留最近5轮

    prompt = f"""你是一个百科对话助手，用于解答用户的各种问题。
对话历史：
{history}

长期记忆（RAG检索结果）：
{context}

用户问题：
{query}

请结合上下文回答："""

    response = llm(prompt)
    return {
        "answer": response,
        "n":state['n']+1
            }

def update_memory(state):
    new_history = state['chat_history']+[f"用户：{state['query']},AI:{state['answer']}"]
    return {'chat_history':new_history}

def update_knowledge(state):
    global index,id_map
    history=state['chat_history'][-5]
    prompt=f"""
以下是最近5轮的对话：
{history}
请根据以上内容总结用户的偏好或某些补充的知识等信息，请回答："""
    response=llm(prompt)
    embedding=embed_model.encode(response)
    embedding = np.array([embedding]).astype('float32') 
    index.add(embedding)
    id_map[len(id_map)]={"id":str(uuid.uuid4()),"text":response}
    return {'knowledge':"0"}

def decide(state):
    if state['n']>0 and state['n']%5==0:
        return "update_knowledge"
    else:
        return "end"
    
def end_node(state):
    return {}
workflow=StateGraph(MemoryState)
workflow.add_node("retrieve",retrieve_node)
workflow.add_node("llm",llm_node)
workflow.add_node("update_memory",update_memory)
workflow.add_node("update_knowledge",update_knowledge)
workflow.add_node("end",end_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve","llm")
workflow.add_edge("llm","update_memory")
workflow.add_conditional_edges(
    "update_memory",
    decide,
    {
        "update_knowledge":"update_knowledge",
        "end":END
     }
)
workflow.add_edge("update_knowledge",END)
memory = MemorySaver()
graph=workflow.compile(checkpointer=memory)



state = {
    "n": 0,
    "query": "",
    "chat_history": [],
    "knowledge": [],
    "answer": ""
}
config1 = {"configurable": {"thread_id": "1"}}
while 1:
    query=input("用户：")
    if query in ["exit", "quit"]:
        break

    state["query"] = query
    state=graph.invoke(state,config=config1)
    print("AI:", state["answer"])

faiss.write_index(index, "docs.faiss")
np.save("id_map.npy", id_map, allow_pickle=True)
model.close()