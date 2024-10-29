!pip install transformers datasets
from google.colab import drive
drive.mount('/content/drive')
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_from_disk

# 加载分词后的数据集
tokenized_train_data = load_from_disk('/content/drive/MyDrive/tokenized_wikitext_train')
tokenized_valid_data = load_from_disk('/content/drive/MyDrive/tokenized_wikitext_valid')

# 加载GPT-2模型
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/gpt2-finetuned-wikitext",
    evaluation_strategy = "steps",
    eval_steps = 500,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy = "steps",
    save_steps = 500,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_valid_data
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("/content/drive/MyDrive/gpt2-finetuned-wikitext")