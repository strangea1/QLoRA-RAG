from datasets import load_dataset

ds = load_dataset("xuqinyang/BaiduBaike-5.63M")

ds.save_to_disk("./dataset")