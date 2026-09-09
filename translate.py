import threading

import torch
from accelerate.utils import get_max_memory
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "TildeAI/TildeOpen-30b"
GPU_MAX_MEMORY = "22GiB"
OFFLOAD_FOLDER = "./offload"

# The model is very large. Keep exactly one tokenizer/model instance per
# Python process and initialize it only when the first translation is needed.
_tokenizer = None
_model = None
_model_lock = threading.Lock()


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This application expects a CUDA-enabled "
            "PyTorch installation and an NVIDIA GPU."
        )

    print("Loading TildeOpen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    print("Preparing 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    max_memory = get_max_memory()
    max_memory[0] = GPU_MAX_MEMORY

    print("Loading TildeOpen-30b in 4-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=OFFLOAD_FOLDER,
        offload_state_dict=True,
    )

    model.eval()

    print("Translation model loaded.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")
    print(f"CUDA reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GiB")

    if hasattr(model, "get_memory_footprint"):
        print(f"Model footprint: {model.get_memory_footprint() / 1024**3:.2f} GiB")

    if hasattr(model, "hf_device_map"):
        devices = sorted({str(device) for device in model.hf_device_map.values()})
        print(f"Model devices: {', '.join(devices)}")

    return tokenizer, model


def get_translation_model():
    """Return the shared tokenizer/model, loading them once when first needed."""
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        # Prevent two simultaneous Flask requests from loading two 30B models.
        with _model_lock:
            if _tokenizer is None or _model is None:
                _tokenizer, _model = load_model()

    return _tokenizer, _model


def get_input_device(model):
    """Return the device hosting the model's input embedding layer."""
    return model.get_input_embeddings().weight.device


def translate_to_latvian(text: str, tokenizer=None, model=None) -> str:
    """Translate one English text to Latvian using the shared TildeOpen model."""
    if not text or not text.strip():
        return text

    # app.py can simply call translate_to_latvian(text). Explicit tokenizer/model
    # arguments are still supported for standalone tests and backwards compatibility.
    if tokenizer is None or model is None:
        tokenizer, model = get_translation_model()

    prompt = f"""
Translate the following text from English to Latvian.

Requirements:
- Preserve the original meaning.
- Use natural, fluent Latvian.
- Keep trivia names, titles, numbers, dates, and proper nouns accurate.
- Do not answer the trivia question.
- Do not add explanations.
- Do not summarize.
- Return only the Latvian translation.

English text:
{text}

Latvian translation:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_device = get_input_device(model)
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            repetition_penalty=1.2,
            do_sample=False,
        )

    prompt_token_count = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_token_count:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return result.strip()


def main():
    sample = "Which planet is known as the Red Planet?"
    print("English:", sample)
    print("Latvian:", translate_to_latvian(sample))


if __name__ == "__main__":
    main()
