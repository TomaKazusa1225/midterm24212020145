from IPython import get_ipython
from IPython.display import display
# 挂载Google Drive
#from google.colab import drive
#drive.mount('/content/drive')

# 安装依赖库
#!#
pip install transformers datasets

from datasets import load_dataset
from transformers import GPT2Tokenizer

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    tokenized_output = tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    tokenized_output["labels"] = tokenized_output["input_ids"].clone()
    return tokenized_output

# 对训练集和验证集进行分词处理
tokenized_train_data = dataset['train'].map(tokenize_function, batched=True)
tokenized_valid_data = dataset['validation'].map(tokenize_function, batched=True)

# 保存分词后的数据集到Google Drive
tokenized_train_data.save_to_disk('/content/drive/MyDrive/tokenized_wikitext_train')
tokenized_valid_data.save_to_disk('/content/drive/MyDrive/tokenized_wikitext_valid')

# 打印部分处理后的训练数据用于确认
print(tokenized_train_data[0])