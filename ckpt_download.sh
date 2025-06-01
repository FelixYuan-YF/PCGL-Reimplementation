export HF_ENDPOINT=https://hf-mirror.com
mkdir -p LLM

huggingface-cli download --resume-download Qwen/Qwen2.5-3B --local-dir LLM/Qwen2.5-3B

huggingface-cli download --resume-download Qwen/Qwen2.5-3B-Instruct --local-dir LLM/Qwen2.5-3B-Instruct