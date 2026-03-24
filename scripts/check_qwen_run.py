from beacon_system.llm.client import LLMClient
from beacon_system.llm.config import ModelConfig

cfg = ModelConfig(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    model_name="qwen/qwen3-32b",
)

client = LLMClient(cfg)
text = client.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Reply exactly: OpenRouter Qwen test passed."}
])

print(text)