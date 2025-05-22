import requests
from difflib import unified_diff
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

# Функция для обращения к локальной Ollama (модель gemma3:27B)
def ollama_llm(prompt: str, model: str = "gemma3:27b") -> str:
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    return result["response"]

# Функция для обращения к Azure LLM (chat/completions)
def azure_llm(prompt: str, system_prompt: str = (
    "You are a code assistant. "
    "Your job is to generate a unified diff (patch format) for the given Python function, "
    "according to the user's instruction. "
    "Do not delete the function unless explicitly asked. "
    "If no changes are needed, return an empty diff. "
    "Do not add explanations, markdown, or comments. Only the diff.\n"
    "Example:\n"
    "--- a/example.py\n"
    "+++ b/example.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def foo():\n"
    "+    print('hello')\n"
    "     pass\n"
), max_tokens: int = 800, temperature: float = 1.0) -> str:
    endpoint = os.getenv("AZURE_LLM_ENDPOINT")
    api_version = os.getenv("AZURE_LLM_API_VERSION")
    api_key = os.getenv("AZURE_LLM_API_KEY")
    deployment = os.getenv("AZURE_LLM_MODEL")
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    print("[DEBUG] Azure LLM HTTP POST:")
    print(f"  endpoint: {endpoint}")
    print(f"  api_version: {api_version}")
    print(f"  deployment: {deployment}")
    print(f"  headers: {{'api-key': '***', 'Content-Type': 'application/json'}}")
    print(f"  messages: {json.dumps(messages, ensure_ascii=False, indent=2)[:2000]}\n...")
    print(f"  max_completion_tokens: {max_tokens}")
    print(f"  temperature: {temperature}")
    response = client.chat.completions.create(
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )
    print("[DEBUG] Оригинальный response от Azure LLM:", response)
    return response.choices[0].message.content

class CodeAgent:
    def __init__(self, llm_callable):
        self.llm = llm_callable
        # Загружаем шаблон промпта из файла
        prompt_path = os.path.join(os.path.dirname(__file__), "ilya.prompt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def make_prompt(self, instruction: str, code: str, metadata: dict) -> str:
        return self.prompt_template.format(
            instruction=instruction,
            filepath=metadata.get('filepath', ''),
            obj_type=metadata.get('type', ''),
            obj_name=metadata.get('name', ''),
            code=code
        )

    def apply_instruction(self, instruction: str, code: str, metadata: dict) -> str:
        prompt = self.make_prompt(instruction, code, metadata)
        print(f"[DEBUG] Промпт для LLM (первые 1000 символов):\n{prompt[:1000]}...\n")
        result = self.llm(code)
        return result

    def generate_diff(self, original: str, modified: str) -> str:
        diff = unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile='original.py',
            tofile='modified.py'
        )
        return '\n'.join(diff) 