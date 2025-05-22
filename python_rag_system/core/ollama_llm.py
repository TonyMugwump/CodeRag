import requests
import json
from typing import Dict, List, Optional, Any
from rich.console import Console

console = Console()

class OllamaLLM:
    """Клиент для работы с Ollama LLM"""
    
    def __init__(self, 
                 model: str = "gemma3:27b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        
        # Проверяем доступность Ollama
        self._check_ollama_availability()
        console.print(f"[green]Подключен к Ollama модели: {model}[/green]")
    
    def _check_ollama_availability(self):
        """Проверяет доступность Ollama сервера и модели"""
        try:
            # Проверяем сервер
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama API недоступен: {response.status_code}")
            
            # Проверяем модель
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                console.print(f"[yellow]Предупреждение: модель {self.model} не найдена в Ollama[/yellow]")
                console.print(f"[yellow]Доступные модели: {', '.join(model_names)}[/yellow]")
                console.print(f"[yellow]Попробуйте: ollama pull {self.model}[/yellow]")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Не удается подключиться к Ollama: {e}")
    
    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.3) -> str:
        """Генерирует ответ на промпт"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Увеличиваем таймаут для больших моделей
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                console.print(f"[red]Ошибка Ollama API: {response.status_code}[/red]")
                console.print(f"[red]Ответ: {response.text}[/red]")
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Ошибка запроса к Ollama: {e}[/red]")
            raise
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Чат с моделью (поддержка диалогов)"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                console.print(f"[red]Ошибка Ollama Chat API: {response.status_code}[/red]")
                console.print(f"[red]Ответ: {response.text}[/red]")
                raise Exception(f"Ollama Chat API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Ошибка запроса к Ollama Chat: {e}[/red]")
            raise
    
    def analyze_code(self, code: str, analysis_type: str = "general") -> str:
        """Анализирует код с помощью LLM"""
        
        system_prompts = {
            "general": "Ты эксперт по анализу Python кода. Анализируй код детально и предоставляй полезные инсайты о функциональности, сложности и качестве кода.",
            "performance": "Ты эксперт по производительности Python кода. Анализируй код с точки зрения производительности и предлагай оптимизации.",
            "security": "Ты эксперт по безопасности Python кода. Ищи потенциальные уязвимости и проблемы безопасности.",
            "architecture": "Ты архитектор ПО. Анализируй архитектурные решения в коде и предлагай улучшения."
        }
        
        system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        
        prompt = f"""
Проанализируй следующий Python код:

```python
{code}
```

Предоставь детальный анализ включающий:
1. Назначение и функциональность
2. Сложность и читаемость
3. Потенциальные проблемы
4. Рекомендации по улучшению
5. Оценка качества кода (1-10)

Отвечай на русском языке.
"""
        
        return self.generate(prompt, system_prompt)
    
    def explain_flow(self, flow_description: str, context: str = "") -> str:
        """Объясняет поток выполнения кода"""
        
        system_prompt = "Ты эксперт по архитектуре ПО. Объясняй flow кода понятно и структурированно, создавай диаграммы в текстовом виде."
        
        prompt = f"""
Объясни поток выполнения следующего кода:

{flow_description}

Дополнительный контекст:
{context}

Предоставь:
1. Пошаговое объяснение выполнения
2. Диаграмму потока в текстовом виде
3. Ключевые точки принятия решений
4. Потенциальные проблемы в потоке
5. Рекомендации по улучшению архитектуры

Отвечай на русском языке.
"""
        
        return self.generate(prompt, system_prompt)
    
    def answer_question(self, question: str, code_context: str) -> str:
        """Отвечает на вопросы о коде"""
        
        system_prompt = "Ты эксперт программист. Отвечай на вопросы о коде основываясь на предоставленном контексте. Будь точным и конкретным."
        
        prompt = f"""
Вопрос: {question}

Контекст кода:
{code_context}

Предоставь детальный ответ основываясь на анализе предоставленного кода. Включи примеры кода если необходимо.

Отвечай на русском языке.
"""
        
        return self.generate(prompt, system_prompt)
    
    def suggest_improvements(self, code: str, complexity_info: str = "") -> str:
        """Предлагает улучшения кода"""
        
        system_prompt = "Ты senior разработчик. Анализируй код и предлагай конкретные улучшения по производительности, читаемости и архитектуре."
        
        prompt = f"""
Проанализируй следующий код и предложи улучшения:

```python
{code}
```

Информация о сложности:
{complexity_info}

Предоставь:
1. Конкретные предложения по улучшению
2. Примеры улучшенного кода
3. Объяснение преимуществ каждого улучшения
4. Приоритизацию изменений (критичные, важные, желательные)
5. Потенциальные риски изменений

Отвечай на русском языке.
"""
        
        return self.generate(prompt, system_prompt)
    
    def find_similarities(self, main_code: str, similar_codes: List[str]) -> str:
        """Анализирует сходства между функциями"""
        
        system_prompt = "Ты эксперт по анализу кода. Находи паттерны, сходства и различия между функциями."
        
        similar_codes_text = "\n\n".join([f"Функция {i+1}:\n```python\n{code}\n```" for i, code in enumerate(similar_codes)])
        
        prompt = f"""
Проанализируй сходства между основной функцией и похожими функциями:

Основная функция:
```python
{main_code}
```

Похожие функции:
{similar_codes_text}

Предоставь:
1. Общие паттерны и подходы
2. Ключевые различия в реализации
3. Лучшие практики из анализируемых функций
4. Рекомендации по унификации или рефакторингу
5. Потенциальные возможности для создания общих утилит

Отвечай на русском языке.
"""
        
        return self.generate(prompt, system_prompt)
    
    def get_available_models(self) -> List[str]:
        """Получает список доступных моделей"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return [] 