import torch
from gpt2model import GPTModel, generate, text_to_token_ids, token_ids_to_text
from bpe_openai_gpt2 import get_encoder

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

CHOOSE_MODEL = "gpt2-small (124M)"

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model = GPTModel(BASE_CONFIG)

model.load_state_dict(torch.load("gpt2-small124M-sft.pth"))
model.eval()

# Uses NVIDIA GPU if available (only works for NVIDIA), otherwise uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    user_input = input("> ")

    tokenizer = get_encoder(model_name="gpt2_tokenizer", models_dir=".")

    if user_input.lower() == "exit":
        print("Graceful Exit")
        break

    prompt = (
            f"### input:\n{user_input}\n"
            f"### output:\n"
            )
    
    token_ids = generate(
            model = model,
            idx=text_to_token_ids(prompt, tokenizer),
            max_new_tokens=500,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
            )

    formatted_output = token_ids_to_text(token_ids, tokenizer)

    output = formatted_output.split("### output:")[-1].strip()
    
    print(output)
