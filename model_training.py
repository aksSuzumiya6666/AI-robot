# model_training.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

def load_text(file_path):
    """加载文本数据"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def prepare_dataset(text):
    """将文本转换为 Hugging Face Dataset 格式"""
    return Dataset.from_dict({"text": [text]})

def tokenize_data(dataset, tokenizer):
    """对数据进行分词处理"""
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    return dataset.map(tokenize_function, batched=True)

def format_labels(tokenized_dataset):
    """为模型创建标签"""
    def add_labels(examples):
        input_ids = torch.tensor(examples['input_ids'])
        return {'labels': input_ids}

    return tokenized_dataset.map(add_labels, batched=True)

def create_training_args():
    """创建训练参数"""
    return TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=2,
        num_train_epochs=1000,  # 设置训练轮次
        learning_rate=5e-5,  # 设置初始学习率
        logging_dir='./logs',
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,  # 启用混合精度训练
        lr_scheduler_type='cosine',  # 使用余弦退火学习率调度
        warmup_steps=1000,  # 设置学习率预热步数
    )

def train_model(data_file, model_save_path, tokenizer_save_path):
    """训练 GPT-2 模型"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')  # 使用更大的预训练模型
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    text = load_text(data_file)
    dataset = prepare_dataset(text)
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    tokenized_dataset = format_labels(tokenized_dataset)

    training_args = create_training_args()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

if __name__ == "__main__":
    data_file = 'data1.txt'  #要训练的内容
    model_save_path = './gpt2_trained_model' #模型保存的目录
    tokenizer_save_path = './tokenizer' #保存的目录
    train_model(data_file, model_save_path, tokenizer_save_path)
