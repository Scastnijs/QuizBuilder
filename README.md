# QuizBuilder
Quiz builder with Open Trivia API https://opentdb.com/
<br/>
Requirements: python installation + tool to install libraries, in this example pip is used <br/>
How to run (terminal): <br/>
<br/>
1)clone to local machine <br/>
git clone https://github.com/Scastnijs/QuizBuilder.git
<br/>
<br/>
2) Download TildeOpen-30b https://huggingface.co/TildeAI/TildeOpen-30b <br/>
~60GB <br/>
<br/>
3) PyTorch 2.14.0 + CUDA 13.0, including Python 3.12
<br/>
download Python 3.12 https://www.python.org/downloads/release/python-3120/
<br/>
create virtual environment <br/>
py -3.12 -m venv my_env
<br/>
<br/>
4)activate virtual environment <br/>
.\my_env\Scripts\Activate.ps1
<br/>
5)install libraries <br/>
python -m pip install flask requests
<br/>
!!! GPU based torch (CPU based will be too slow) :
<br/>
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
<br/>
python -m pip install transformers accelerate sentencepiece safetensors tiktoken protobuf
<br/>
python -m pip install --upgrade bitsandbytes accelerate
<br/>
<br/>
python -c "import torch; print('BF16 supported:', torch.cuda.is_bf16_supported())"
<br/>
[should be] BF16 supported: True
<br/>
<br/>
5*) restart of python kernel may be required <br/>
<br/>
6)run application <br/>
python app.py
<br/>
<br/>
7)open in browser <br/>
http://127.0.0.1:5000
<br/>
7*) Ctrl+c to close <br/>
<br/>
<br/>
8)deactivate virtual environment, when you are finished<br/>
deactivate
<br/>
