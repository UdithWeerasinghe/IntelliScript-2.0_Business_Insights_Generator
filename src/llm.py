import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import json

def load_config(config_path='config/config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def authenticate_and_load_model(config):
    hf_api_key = config.get("hf_api_key")
    if hf_api_key:
        login(hf_api_key)
    model_name = config.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
    device = config.get("device", "cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, device

def generate_response(system_prompt, user_prompt, temperature=0.7, max_tokens=500, model=None, tokenizer=None, device="cuda"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    model.config.pad_token_id = model.config.eos_token_id
    model_inputs['attention_mask'] = model_inputs['input_ids'].ne(model.config.pad_token_id).long()
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs['attention_mask'],
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=10,
        top_p=0.9,
    )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response
