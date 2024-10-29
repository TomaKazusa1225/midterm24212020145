from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 加载微调后的模型和分词器
model = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/gpt2-finetuned-wikitext')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 设置生成的参数
def generate_sentences(prompt, num_sentences=10):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=100,  # 控制生成的最大长度
        num_return_sequences=num_sentences,  # 生成的句子数量
        no_repeat_ngram_size=2,  # 防止重复
        repetition_penalty=1.2,  #1
        top_k=40,         #2
        top_p=0.92,  # nucleus sampling
        temperature=0.7,  # 控制生成的多样性
        do_sample=True
    )
    # 解码并打印生成的句子
    sentences = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    return sentences

# 给定提示词，生成句子
prompt = "Once upon a time"
generated_sentences = generate_sentences(prompt)
for idx, sentence in enumerate(generated_sentences):
    print(f"Sentence {idx + 1}: {sentence}")
