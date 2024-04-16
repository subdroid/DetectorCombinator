from transformers import AutoTokenizer

model_list = ["facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B"]

for model in model_list:
    # Load GPT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model,cache_dir="transformers_cache")

    # Input text
    input_text = "Elementary, my dear Watson."

    # Tokenize input text
    tokenized_input = tokenizer.encode(input_text, add_special_tokens=False)

    # Decode tokenized input to subwords
    subwords = tokenizer.convert_ids_to_tokens(tokenized_input)

    # Print subwords
    print(f"{model} \t {len(subwords)} \t Subwords: {subwords}")