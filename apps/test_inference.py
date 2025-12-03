import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Load tokenizer and model
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    # Move model to device (XPU if supported, fallback to CPU)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model.to(device)

    # Prepare prompt (Mistral uses [INST] ... [/INST] format)
    prompt = "<s>[INST] Explain quantum computing in simple terms. [/INST]"

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)

    # Decode and print
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    main()
