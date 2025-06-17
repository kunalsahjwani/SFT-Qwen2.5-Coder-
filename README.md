#  Qwen2.5-Coder  Fine-Tuned for Streamlit Code Generation (QLoRA)

This project fine-tunes the `Qwen2.5-Coder-1.5B-Instruct` model to generate Streamlit applications from natural language prompts, using memory-efficient QLoRA on free-tier Google Colab GPUs.

---

##  What This Is

- Fine-tuning using **QLoRA** (4-bit quantization with LoRA adapters)
- Target task: **Streamlit code generation**
- Uses synthetic examples generated from structured prompts
- Training and inference run entirely on **free Colab GPU** instances

---

##  Model Details

- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Method**: QLoRA (via bitsandbytes) + PEFT (LoRA)
- **Dataset**: 50 synthetic instruction-output pairs for Streamlit apps
- **Training Format**: Instruction-following (Qwen chat-style)

---

##  Training Tips

>  This tutorial is built to **run on free Colab GPUs**.  
> To reduce loss and improve model quality, **adjust training parameters** (e.g., epochs, learning rate, batch size) as recommended in the code comments.
> To run this tutorial you need a huggingface account and an api key


---
##  Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("kunalsahjwani/qwen-streamlit-coder")
model = AutoModelForCausalLM.from_pretrained("kunalsahjwani/qwen-streamlit-coder")

messages = [
    {"role": "system", "content": "You are Qwen, a helpful coding assistant. Generate clean, working code."},
    {"role": "user", "content": "Create a Streamlit app: personal finance tracker with charts"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=400)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

##  Files Included

- `qwen-streamlit-coder/`: LoRA fine-tuned model directory
- `README.md`: This file
- `training_script.ipynb`: Full Colab notebook (coming soon)



---

**Author**: [Kunal Sahjwani](https://www.linkedin.com/in/kunalsahjwani)  
Model available on [Hugging Face](https://huggingface.co/kunalsahjwani/qwen-streamlit-coder)
