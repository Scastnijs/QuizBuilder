import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "TildeAI/TildeOpen-30b"

def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Model loaded.")

    return tokenizer, model

def translate_to_latvian(
    text: str,
    tokenizer,
    model
) -> str:

    prompt = f"""
Translate the following text from English to Latvian.

Requirements:
- Preserve the original meaning.
- Use natural, fluent Latvian.
- Use professional language.
- Do not add explanations.
- Do not summarize.
- Return only the Latvian translation.

English text:
{text}

Latvian translation:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            repetition_penalty=1.2,
            do_sample=False
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove the original prompt from the generated result
    if result.startswith(prompt):
        result = result[len(prompt):]

    return result.strip()

def main():

    tokenizer, model = load_model()

    english_text = """
I have several years of experience working as a software developer
and data engineer. My responsibilities included developing data
processing pipelines, maintaining databases, and supporting
production systems.
"""

    translation = translate_to_latvian(
        english_text,
        tokenizer,
        model
    )

    print("\n--- English ---")
    print(english_text)

    print("\n--- Latvian ---")
    print(translation)


if __name__ == "__main__":
    main()