import os
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from dotenv import load_dotenv

from .rag_engine import PythonRAGEngine
from .ollama_llm import OllamaLLM

load_dotenv()
console = Console()

class UnifiedLLMCodeAnalyzer:
    """Унифицированный анализатор кода с поддержкой OpenAI и Ollama"""
    
    def __init__(self, 
                 rag_engine: PythonRAGEngine, 
                 llm_provider: str = "ollama",
                 model: str = None,
                 ollama_base_url: str = "http://localhost:11434"):
        
        self.rag_engine = rag_engine
        self.llm_provider = llm_provider.lower()
        
        if self.llm_provider == "ollama":
            default_model = model or "gemma3:27b"
            try:
                self.llm = OllamaLLM(model=default_model, base_url=ollama_base_url)
                self.available = True
                console.print(f"[green]Используется Ollama LLM: {default_model}[/green]")
            except Exception as e:
                console.print(f"[red]Ошибка подключения к Ollama: {e}[/red]")
                self.available = False
                
        elif self.llm_provider == "openai":
            try:
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                if not openai.api_key:
                    raise Exception("OPENAI_API_KEY не установлен")
                
                self.openai = openai
                self.model = model or "gpt-4"
                self.available = True
                console.print(f"[green]Используется OpenAI: {self.model}[/green]")
            except Exception as e:
                console.print(f"[red]Ошибка подключения к OpenAI: {e}[/red]")
                self.available = False
        else:
            console.print(f"[red]Неподдерживаемый провайдер LLM: {llm_provider}[/red]")
            self.available = False
    
    def analyze_function(self, function_name: str) -> Dict[str, Any]:
        """Анализирует функцию с помощью LLM"""
        
        if not self.available:
            return {"error": "LLM недоступен"}
        
        # Получаем контекст функции из RAG
        context = self.rag_engine.get_function_context(function_name)
        
        if "error" in context:
            return context
        
        try:
            main_function = context['main_function']
            code = main_function['document']
            
            if self.llm_provider == "ollama":
                analysis = self.llm.analyze_code(code, "general")
            else:  # OpenAI
                prompt = self._create_function_analysis_prompt(context)
                analysis = self._call_openai(prompt, "Ты эксперт по анализу Python кода.")
            
            return {
                "function_name": function_name,
                "context": context,
                "llm_analysis": analysis,
                "provider": self.llm_provider
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM анализа: {e}[/red]")
            return {"error": f"Ошибка LLM анализа: {e}"}
    
    def explain_code_flow(self, entry_function: str) -> Dict[str, Any]:
        """Объясняет flow кода начиная с entry функции"""
        
        if not self.available:
            return {"error": "LLM недоступен"}
        
        # Получаем контекст функции
        context = self.rag_engine.get_function_context(entry_function)
        
        if "error" in context:
            return context
        
        try:
            # Строим описание flow
            flow_description = self._build_flow_description(context)
            
            if self.llm_provider == "ollama":
                explanation = self.llm.explain_flow(flow_description)
            else:  # OpenAI
                prompt = self._create_flow_explanation_prompt(context, flow_description)
                explanation = self._call_openai(prompt, "Ты эксперт по архитектуре ПО.")
            
            return {
                "entry_function": entry_function,
                "flow_description": flow_description,
                "llm_explanation": explanation,
                "context": context,
                "provider": self.llm_provider
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM анализа flow: {e}[/red]")
            return {"error": f"Ошибка LLM анализа flow: {e}"}
    
    def answer_code_question(self, question: str, context_search: str = None) -> Dict[str, Any]:
        """Отвечает на вопросы о коде используя RAG"""
        
        if not self.available:
            return {"error": "LLM недоступен"}
        
        # Если не указан контекст для поиска, используем сам вопрос
        search_query = context_search or question
        
        # Ищем релевантный код
        search_results = self.rag_engine.search(search_query, n_results=5)
        
        if not search_results:
            return {"error": "Не найден релевантный код для ответа на вопрос"}
        
        try:
            # Формируем контекст из результатов поиска
            code_context = self._format_search_results(search_results)
            
            if self.llm_provider == "ollama":
                answer = self.llm.answer_question(question, code_context)
            else:  # OpenAI
                prompt = self._create_question_answer_prompt(question, search_results)
                answer = self._call_openai(prompt, "Ты эксперт программист.")
            
            return {
                "question": question,
                "answer": answer,
                "context_used": search_results,
                "search_query": search_query,
                "provider": self.llm_provider
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM ответа: {e}[/red]")
            return {"error": f"Ошибка LLM ответа: {e}"}
    
    def suggest_improvements(self, function_name: str) -> Dict[str, Any]:
        """Предлагает улучшения для функции"""
        
        if not self.available:
            return {"error": "LLM недоступен"}
        
        context = self.rag_engine.get_function_context(function_name)
        
        if "error" in context:
            return context
        
        try:
            main_function = context['main_function']
            code = main_function['document']
            complexity_info = f"Сложность: {main_function['metadata'].get('complexity', 'N/A')}"
            
            if self.llm_provider == "ollama":
                suggestions = self.llm.suggest_improvements(code, complexity_info)
            else:  # OpenAI
                prompt = self._create_improvement_prompt(context)
                suggestions = self._call_openai(prompt, "Ты senior разработчик.")
            
            return {
                "function_name": function_name,
                "context": context,
                "suggestions": suggestions,
                "complexity_score": main_function['metadata'].get('complexity', 0),
                "provider": self.llm_provider
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM предложений: {e}[/red]")
            return {"error": f"Ошибка LLM предложений: {e}"}
    
    def find_similar_functions(self, function_name: str) -> Dict[str, Any]:
        """Находит похожие функции"""
        
        if not self.available:
            return {"error": "LLM недоступен"}
        
        # Получаем контекст функции
        context = self.rag_engine.get_function_context(function_name)
        
        if "error" in context:
            return context
        
        # Ищем похожие функции
        similar_results = self.rag_engine.search(
            f"function similar to {function_name}", 
            n_results=10,
            filter_type="function"
        )
        
        # Фильтруем саму функцию из результатов
        similar_functions = [
            result for result in similar_results 
            if result['metadata']['name'] != function_name
        ][:5]
        
        if not similar_functions:
            return {"message": f"Не найдено похожих функций для {function_name}"}
        
        try:
            main_function = context['main_function']
            main_code = main_function['document']
            similar_codes = [func['document'] for func in similar_functions]
            
            if self.llm_provider == "ollama":
                similarity_analysis = self.llm.find_similarities(main_code, similar_codes)
            else:  # OpenAI
                prompt = self._create_similarity_analysis_prompt(main_function, similar_functions)
                similarity_analysis = self._call_openai(prompt, "Ты эксперт по анализу кода.")
            
            return {
                "function_name": function_name,
                "similar_functions": similar_functions,
                "similarity_analysis": similarity_analysis,
                "provider": self.llm_provider
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка анализа похожих функций: {e}[/red]")
            return {"error": f"Ошибка анализа похожих функций: {e}"}
    
    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """Вызывает OpenAI API"""
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _build_flow_description(self, context: Dict[str, Any]) -> str:
        """Строит описание flow функции"""
        main_function = context['main_function']
        flow_info = context.get('flow_info', {})
        
        description = f"Функция: {main_function['metadata']['name']}\n"
        description += f"Файл: {main_function['metadata']['file_path']}\n"
        description += f"Сложность: {main_function['metadata'].get('complexity', 'N/A')}\n\n"
        description += f"Код функции:\n{main_function['document']}\n\n"
        
        if flow_info.get('calls'):
            description += f"Вызывает функции: {', '.join([c.split('::')[-1] for c in flow_info['calls']])}\n"
        
        if flow_info.get('called_by'):
            description += f"Вызывается из: {', '.join([c.split('::')[-1] for c in flow_info['called_by']])}\n"
        
        return description
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Форматирует результаты поиска для LLM"""
        context = ""
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            context += f"\n--- Результат {i} ---\n"
            context += f"Имя: {metadata['name']}\n"
            context += f"Тип: {metadata['type']}\n"
            context += f"Файл: {metadata['file_path']}\n"
            context += f"Код:\n{result['document']}\n"
        
        return context
    
    # Методы для создания промптов (совместимость с OpenAI)
    def _create_function_analysis_prompt(self, context: Dict[str, Any]) -> str:
        main_function = context['main_function']
        return f"""
Проанализируй следующую Python функцию:

Имя: {main_function['metadata']['name']}
Файл: {main_function['metadata']['file_path']}
Сложность: {main_function['metadata'].get('complexity', 'N/A')}

Код:
{main_function['document']}

Предоставь детальный анализ включающий назначение, сложность, потенциальные проблемы и рекомендации.
"""
    
    def _create_flow_explanation_prompt(self, context: Dict[str, Any], flow_description: str) -> str:
        return f"""
Объясни поток выполнения следующего кода:

{flow_description}

Предоставь пошаговое объяснение и диаграмму потока.
"""
    
    def _create_question_answer_prompt(self, question: str, search_results: List[Dict[str, Any]]) -> str:
        context = self._format_search_results(search_results)
        return f"""
Вопрос: {question}

Контекст кода:
{context}

Ответь на вопрос основываясь на предоставленном коде.
"""
    
    def _create_improvement_prompt(self, context: Dict[str, Any]) -> str:
        main_function = context['main_function']
        return f"""
Предложи улучшения для следующей функции:

{main_function['document']}

Сложность: {main_function['metadata'].get('complexity', 'N/A')}

Предоставь конкретные предложения по улучшению.
"""
    
    def _create_similarity_analysis_prompt(self, main_function: Dict[str, Any], similar_functions: List[Dict[str, Any]]) -> str:
        similar_codes = [func['document'] for func in similar_functions]
        return f"""
Проанализируй сходства между функциями:

Основная функция:
{main_function['document']}

Похожие функции:
{chr(10).join([f"Функция {i+1}: {code}" for i, code in enumerate(similar_codes)])}

Найди общие паттерны и различия.
"""
    
    def display_analysis_result(self, result: Dict[str, Any], analysis_type: str = "function"):
        """Отображает результат анализа"""
        
        if "error" in result:
            console.print(f"[red]Ошибка: {result['error']}[/red]")
            return
        
        provider_info = f" ({result.get('provider', 'unknown')})" if 'provider' in result else ""
        
        if analysis_type == "function":
            self._display_function_analysis(result, provider_info)
        elif analysis_type == "flow":
            self._display_flow_analysis(result, provider_info)
        elif analysis_type == "question":
            self._display_question_answer(result, provider_info)
        elif analysis_type == "improvements":
            self._display_improvements(result, provider_info)
        elif analysis_type == "similar":
            self._display_similar_functions(result, provider_info)
    
    def _display_function_analysis(self, result: Dict[str, Any], provider_info: str):
        console.print(Panel(
            result['llm_analysis'],
            title=f"🧠 Анализ функции: {result['function_name']}{provider_info}",
            border_style="blue"
        ))
    
    def _display_flow_analysis(self, result: Dict[str, Any], provider_info: str):
        console.print(Panel(
            result['llm_explanation'],
            title=f"🌊 Анализ flow: {result['entry_function']}{provider_info}",
            border_style="green"
        ))
    
    def _display_question_answer(self, result: Dict[str, Any], provider_info: str):
        console.print(Panel(
            f"**Вопрос:** {result['question']}\n\n**Ответ:**\n{result['answer']}",
            title=f"❓ Ответ на вопрос{provider_info}",
            border_style="yellow"
        ))
    
    def _display_improvements(self, result: Dict[str, Any], provider_info: str):
        console.print(Panel(
            result['suggestions'],
            title=f"⚡ Предложения по улучшению: {result['function_name']}{provider_info}",
            border_style="magenta"
        ))
    
    def _display_similar_functions(self, result: Dict[str, Any], provider_info: str):
        if "message" in result:
            console.print(f"[yellow]{result['message']}[/yellow]")
            return
        
        console.print(Panel(
            result['similarity_analysis'],
            title=f"🔗 Анализ похожих функций: {result['function_name']}{provider_info}",
            border_style="cyan"
        )) 