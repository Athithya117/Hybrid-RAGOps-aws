from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from ctransformers import AutoModelForCausalLM

app = FastAPI()

@serve.deployment(route_prefix="/v1")
@serve.ingress(app)
class LLMServer:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "/models",
            model_file="Qwen3-4B-Q2_K.gguf",
            model_type="qwen",
            gpu_layers=0
        )

    @app.post("/chat/completions")
    async def chat(self, body: dict):
        try:
            prompt = body["messages"][-1]["content"]
            out = self.model(prompt, max_new_tokens=256, temperature=0.7)
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": out}}]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

entrypoint = LLMServer.bind()
