from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM,DataCollatorForLanguageModeling, AutoProcessor, AutoTokenizer, BitsAndBytesConfig,get_linear_schedule_with_warmup
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import datasets
import tqdm
import os

model_addr = "./model/Qwen2.5-7B" 

#量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 启用 4bit 量化
    bnb_4bit_use_double_quant=True, # 双重量化，进一步省显存
    bnb_4bit_quant_type="nf4",      # 更好的量化精度
    bnb_4bit_compute_dtype="bfloat16",  # 训练时计算精度
    llm_int8_enable_fp32_cpu_offload=True
)

#量化后的模型加载
model = AutoModelForCausalLM.from_pretrained(
    model_addr, 
    quantization_config=bnb_config,
    device_map="cuda"
)

tokenizer=AutoTokenizer.from_pretrained(model_addr)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# # LoRA参数
peft_config=LoraConfig(
    r=4,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","v_proj"]
)



# LoRA微调

model=get_peft_model(model,peft_config)
ds=datasets.load_from_disk("./dataset")
ds=ds['train']

def format(example):
    return {
        "text":f"""用户：{example['title']}是什么？
助手：{example['summary']}{tokenizer.eos_token}
"""
    }
def tokenize_function(example):
    return tokenizer(example['text'],truncation=True,padding="max_length",max_length=512)

ds = ds.select(range(500))
ds=ds.map(format)
ds=ds.map(tokenize_function,batched=True,remove_columns=ds.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
train_dataloader=DataLoader(ds,batch_size=1,shuffle=True,collate_fn=data_collator)

optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
num_epochs = 3
num_train_step=num_epochs*len(train_dataloader)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=num_train_step
)

model.train()
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for step,batch in enumerate(loop):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        output=model(**batch)
        loss=output.loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        loop.set_postfix(loss=loss.item())

model.save_pretrained("./qwen_qlora_manual")

# 测试

# model=PeftModel.from_pretrained(model,"./qwen_qlora_manual",device="cuda")
# inputs = tokenizer(
#     text="""用户：丈夫是什么？
# 助手：""",
#     return_tensors="pt",
# ).to(model.device)

# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=400,eos_token_id=tokenizer.eos_token_id)
# response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

# print("模型回复：")
# print(response)