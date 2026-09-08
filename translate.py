import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate.utils import get_max_memory


MODEL_NAME = "TildeAI/TildeOpen-30b"

# Leave roughly 2 GiB of the RTX 4090 free for CUDA context,
# temporary tensors, activations, and text generation.
GPU_MAX_MEMORY = "22GiB"

# If Accelerate cannot keep every model component on the GPU,
# it can offload remaining modules to CPU RAM (or, if necessary,
# to this folder on disk).
OFFLOAD_FOLDER = "./offload"


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script expects a CUDA-enabled PyTorch "
            "installation and an NVIDIA GPU."
        )

    print("Loading tokenizer...")

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

    print("Loading model in 4-bit mode...")

    # Start with Accelerate's detected GPU/CPU memory limits, then reserve
    # about 2 GiB of VRAM for CUDA context and generation-time allocations.
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

    print("Model loaded.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")
    print(f"CUDA reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GiB")

    if hasattr(model, "get_memory_footprint"):
        print(f"Model footprint: {model.get_memory_footprint() / 1024**3:.2f} GiB")

    if hasattr(model, "hf_device_map"):
        devices = sorted({str(device) for device in model.hf_device_map.values()})
        print(f"Model devices: {', '.join(devices)}")

    return tokenizer, model


def get_input_device(model):
    """Return the device hosting the model's input embedding layer."""
    return model.get_input_embeddings().weight.device


def translate_to_latvian(
    text: str,
    tokenizer,
    model,
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
        return_tensors="pt",
    )

    # With device_map="auto" the model may span GPU and CPU.
    # Inputs must start on the device that owns the embedding layer.
    input_device = get_input_device(model)
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            repetition_penalty=1.2,
            do_sample=False,
        )

    # Decode only newly generated tokens instead of decoding the prompt
    # and then trying to remove it as a string.
    prompt_token_count = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_token_count:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

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
        model,
    )

    print("\n--- English ---")
    print(english_text.strip())

    print("\n--- Latvian ---")
    print(translation)


if __name__ == "__main__":
    main()
