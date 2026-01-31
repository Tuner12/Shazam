from transformers import BertTokenizer

# 使用本地BERT模型路径
bert_path = '/home/leiwenhui/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

print(tokenizer("left lung", return_tensors="pt").input_ids)
print(tokenizer("left lung </s>", return_tensors="pt").input_ids)
