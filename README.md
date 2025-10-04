Oesepheus
A 1B parameter LLM optimized for 8GB M1 MacBook Air.
What is this?
A small AI model that runs locally on your Mac. Fast, private, doesn't need internet.
Why?

Runs on 8GB RAM (most models need 16GB+)
20-30 tokens/second on M1 Air
Your data stays on your machine
Free to use

Install
bashgit clone https://github.com/yourusername/oesepheus.git
cd oesepheus
pip install -r requirements.txt
python run.py
Usage
pythonfrom oesepheus import Oesepheus

model = Oesepheus("models/oesepheus-q4.gguf")
response = model.generate("Explain recursion simply")
print(response)
Or command line:
bashpython run.py --prompt "Write a haiku about code"
Specs

Size: 1.1B parameters (~700MB file)
RAM: Uses 2-3GB when running
Speed: 20-30 tokens/sec on M1
Context: 2048 tokens

What it's good for

Quick answers
Code help
Writing assistance
Learning tool

What it's NOT good for

Complex reasoning (use GPT-4 for that)
Long documents
Production apps

Tech Stack

Base: TinyLlama 1.1B
Quantization: 4-bit (Q4_K_M)
Inference: llama.cpp with Metal
Language: Python

License
MIT

