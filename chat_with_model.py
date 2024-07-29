from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model_and_tokenizer(model_path, tokenizer_path):
    """加载模型和分词器"""
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def chat_with_model(model, tokenizer):
    """与模型进行对话"""
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        inputs = tokenizer(user_input, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # 调整生成参数以启用采样
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_length=100,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"AI: {response}")

if __name__ == "__main__":
    model_path = './gpt2_trained_model'  # 使用训练好的模型路径
    tokenizer_path = './tokenizer'
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    chat_with_model(model, tokenizer)
