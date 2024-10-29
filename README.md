# midterm24212020145

## 概述
基于 Wikitext 数据集微调 GPT-2 语言模型，使用了google colab进行训练，脚本中的部分文件路径需要更改。

## 文件说明
- **`googletrain.py`**：训练模型的脚本。
- **`generate01.py`**：用于生成文本的脚本。
- **`model.safetensors`**：训练后的模型权重文件。

## 使用方法
1. **文本生成**  
   配置环境，使用以下命令运行 `generate01.py` 进行文本生成：
   ```bash
   python generate01.py
