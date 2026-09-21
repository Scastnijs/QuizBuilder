from safetensors import safe_open

FILE = r"C:\Users\andre\.cache\huggingface\hub\models--TildeAI--TildeOpen-30b\snapshots\faea2654cb208f13c0f84702b82c3113400bdd4a\model-00001-of-00013.safetensors"

with safe_open(FILE, framework="pt", device="cpu") as f:

    print("Metadata:")
    print(f.metadata())

    print("\nTensors:")

    for key in f.keys():
        tensor = f.get_tensor(key)

        print("Shape:", tensor.shape)
        print("Type:", tensor.dtype)
        print("Values:")
        print(tensor)

        break