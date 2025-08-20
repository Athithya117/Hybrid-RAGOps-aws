# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="RedHatAI/Qwen3-0.6B-quantized.w4a16")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("RedHatAI/Qwen3-0.6B-quantized.w4a16")
model = AutoModelForCausalLM.from_pretrained("RedHatAI/Qwen3-0.6B-quantized.w4a16")
messages = [
    {"role": "user", "content": "why the same embedding model should be used in both indexing and in inference in RAG?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

