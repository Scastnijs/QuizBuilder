import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate.utils import get_max_memory


MODEL_NAME = "TildeAI/TildeOpen-30b"
GPU_MAX_MEMORY = "22GiB"
OFFLOAD_FOLDER = "./offload"

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

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    print("Preparing 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading model in 4-bit mode...")
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


def translate_quiz_item(question: str, answers: list[dict]) -> dict:
    """
    Translate one complete multiple-choice quiz item in ONE model generation.

    Each answer dict must contain:
        id: stable id such as A0, A1, ...
        text: normalized/original answer text
        translate: True if the answer should be translated, False if it must be kept

    Returns:
        {
            "question": "...",
            "answers": {"A0": "...", "A1": "...", ...}
        }

    Answers marked translate=False are restored from the original normalized input
    after parsing, so the LLM cannot alter numeric/time/money/date values.
    """
    input_lines = [f"Q\tTRANSLATE\t{question}"]
    original_by_id = {}
    translate_by_id = {}

    for answer in answers:
        answer_id = answer["id"]
        text = answer["text"]
        should_translate = bool(answer["translate"])
        mode = "TRANSLATE" if should_translate else "KEEP"

        original_by_id[answer_id] = text
        translate_by_id[answer_id] = should_translate
        input_lines.append(f"{answer_id}\t{mode}\t{text}")

    payload = "\n".join(input_lines)

    prompt = f"""
You are translating one multiple-choice trivia question from English to Latvian.

Rules:
1. Translate every row marked TRANSLATE into natural, accurate Latvian.
2. Copy every row marked KEEP exactly as supplied. Do not change any character.
3. Do not answer the trivia question and do not decide which answer is correct.
4. Preserve names, titles, numbers, symbols, and factual meaning.
5. Keep every row ID exactly unchanged.
6. Return exactly one output row for every input row, in the same order.
7. Output format must be: ID<TAB>text
8. Do not output markdown, commentary, labels, JSON, or extra lines.

Input rows:
{payload}

Output rows:
"""

    raw = _generate(prompt, max_new_tokens=512)

    parsed = {}
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or "\t" not in line:
            continue
        row_id, text = line.split("\t", 1)
        row_id = row_id.strip()
        text = text.strip()
        if row_id == "Q" or row_id in original_by_id:
            parsed[row_id] = text

    if not parsed.get("Q"):
        raise ValueError(
            "TildeOpen response could not be parsed: translated question row Q is missing. "
            f"Raw response: {raw!r}"
        )

    result_answers = {}
    for answer_id, original_text in original_by_id.items():
        # Deterministic answers are never trusted back from the LLM.
        if not translate_by_id[answer_id]:
            result_answers[answer_id] = original_text
            continue

        translated = parsed.get(answer_id)
        if not translated:
            raise ValueError(
                f"TildeOpen response could not be parsed: answer row {answer_id} is missing. "
                f"Raw response: {raw!r}"
            )
        result_answers[answer_id] = translated

    return {
        "question": parsed["Q"],
        "answers": result_answers,
    }


if __name__ == "__main__":
    sample = translate_quiz_item(
        "What is the capital of Germany?",
        [
            {"id": "A0", "text": "Berlin", "translate": True},
            {"id": "A1", "text": "Munich", "translate": True},
            {"id": "A2", "text": "1990", "translate": False},
            {"id": "A3", "text": "04/07/1776", "translate": False},
        ],
    )
    print(sample)
