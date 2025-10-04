Oesepheus
1B parameter LLM optimized for 8GB M1 MacBook Air

What is this?
Small AI model that runs locally. Fast, private, offline.
Why use it?
✓ Runs on 8GB RAM
✓ 20-30 tokens/second  
✓ Data stays local
✓ No API costs

Installation
bashgit clone https://github.com/yourusername/oesepheus.git
cd oesepheus
pip install -r requirements.txt
python run.py

Usage
Python
pythonfrom oesepheus import Oesepheus

model = Oesepheus("models/oesepheus-q4.gguf")
response = model.generate("Explain recursion")
print(response)
Command Line
bash# Single prompt
python run.py --prompt "Write a haiku"

# Interactive mode
python run.py --interactive

# Adjust creativity
python run.py --prompt "Tell a story" --temperature 0.9

Specifications
Model Size:       1.1B parameters
File Size:        ~700MB
RAM Usage:        2-3GB
Inference Speed:  20-30 tokens/sec (M1)
Context Window:   2048 tokens
Quantization:     4-bit (Q4_K_M)
Platform:         macOS (Apple Silicon)

Good For

Quick questions
Code snippets
Text summaries
Writing help
Learning tool

NOT Good For

Complex reasoning
Long documents (>1000 words)
Production apps


Architecture
Base Model:    TinyLlama 1.1B
Layers:        22 transformer blocks
Optimization:  4-bit quantization + Metal acceleration
Inference:     llama.cpp

Requirements
macOS 11.0+
Apple Silicon (M1/M2/M3)
8GB RAM minimum
Python 3.8+
2GB disk space

Configuration
bash--model         Model file path
--ctx-size      Context length (default: 2048)
--temp          Temperature 0.0-1.0 (default: 0.7)
--max-tokens    Max generation length (default: 256)
--threads       CPU threads (default: 4)

License
MIT

Notes
This is a small model. Don't expect GPT-4 quality.
Good for local/private use, learning, and prototyping.
