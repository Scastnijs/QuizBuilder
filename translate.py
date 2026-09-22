import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging
from accelerate.utils import get_max_memory


MODEL_NAME = "TildeAI/TildeOpen-30b"
GPU_MAX_MEMORY = "22GiB"
OFFLOAD_FOLDER = "./offload"

# Suppress Transformers tqdm progress bars such as "Loading weights: 123/543".
transformers_logging.disable_progress_bar()

_tokenizer = None
_model = None
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script expects a CUDA-enabled PyTorch "
            "installation and an NVIDIA GPU."
        )

    print("Loading TildeOpen-30b weights...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    max_memory = get_max_memory()
    max_memory[0] = GPU_MAX_MEMORY

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

    print("TildeOpen-30b weights loaded.")
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
    """Load TildeOpen once per Flask process and reuse it for all requests."""
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        with _model_lock:
            if _tokenizer is None or _model is None:
                _tokenizer, _model = load_model()

    return _tokenizer, _model


def get_input_device(model):
    """Return the device hosting the model's input embedding layer."""
    return model.get_input_embeddings().weight.device


def _generate(prompt: str, max_new_tokens: int = 512) -> str:
    """Run one deterministic TildeOpen generation using the shared model."""
    tokenizer, model = get_translation_model()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = get_input_device(model)
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    # A single 4090 should execute only one generation at a time.
    with _generation_lock:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.2,
                do_sample=False,
            )

    prompt_token_count = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_token_count:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()



def translate_text(text: str, target_language: str) -> str:
    """Translate one standalone English string to the requested target language."""
    target_languages = {
        "lv": "Latvian",
        "de": "German",
        "ua": "Ukrainian",
        "bg": "Bulgarian",
        "cz": "Czech",
        "ee": "Estonian",
        "fi": "Finnish",
        "fr": "French",
        "hu": "Hungarian",
        "is": "Icelandic",
        "it": "Italian",
        "lt": "Lithuanian",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "se": "Swedish",
        "si": "Slovenian",
        "sk": "Slovak",
        "tr": "Turkish",
        "rs": "Serbian",
        "es": "Spanish",
    }

    language_name = target_languages.get(target_language)
    if language_name is None:
        raise ValueError(
            f"Unsupported target language: {target_language}. "
            f"Supported target languages: {', '.join(target_languages)}"
        )

    prompt = f"""
Translate the following text from English to {language_name}.

Requirements:
- Preserve the original meaning.
- Use natural, fluent {language_name}.
- Do not add explanations.
- Do not summarize.
- Return only the {language_name} translation.

English text:
{text}

{language_name} translation:
"""
    return _generate(prompt, max_new_tokens=256)


def translate_to_latvian(text: str) -> str:
    """Translate one standalone string. Kept for other QuizBuilder uses."""
    prompt = f"""
Translate the following text from English to Latvian.

Requirements:
- Preserve the original meaning.
- Use natural, fluent Latvian.
- Do not add explanations.
- Do not summarize.
- Return only the Latvian translation.

English text:
{text}

Latvian translation:
"""
    return _generate(prompt, max_new_tokens=256)


if __name__ == "__main__":
    sample_question = "What is the capital of Germany?"
    print(translate_to_latvian(sample_question))
