import openai
from typing import Dict, List, Optional, Any
import json
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import os
from dotenv import load_dotenv

from .rag_engine import PythonRAGEngine

load_dotenv()
console = Console()

class LLMCodeAnalyzer:
    """LLM анализатор кода с RAG поддержкой"""
    
    def __init__(self, rag_engine: PythonRAGEngine, model: str = "gpt-4"):
        self.rag_engine = rag_engine
        self.model = model
        
        # Инициализируем OpenAI клиент
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            console.print("[red]Предупреждение: OPENAI_API_KEY не установлен. LLM функции будут недоступны.[/red]")
    
    def analyze_function(self, function_name: str) -> Dict[str, Any]:
        """Анализирует функцию с помощью LLM"""
        
        # Получаем контекст функции из RAG
        context = self.rag_engine.get_function_context(function_name)
        
        if "error" in context:
            return context
        
        # Формируем промпт для LLM
        prompt = self._create_function_analysis_prompt(context)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по анализу Python кода. Анализируй код детально и предоставляй полезные инсайты."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "function_name": function_name,
                "context": context,
                "llm_analysis": analysis,
                "recommendations": self._extract_recommendations(analysis)
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM анализа: {e}[/red]")
            return {"error": f"Ошибка LLM анализа: {e}"}
    
    def explain_code_flow(self, entry_function: str) -> Dict[str, Any]:
        """Объясняет flow кода начиная с entry функции"""
        
        # Получаем контекст функции
        context = self.rag_engine.get_function_context(entry_function)
        
        if "error" in context:
            return context
        
        # Строим граф вызовов
        flow_map = self._build_flow_map(entry_function, max_depth=3)
        
        # Формируем промпт для объяснения flow
        prompt = self._create_flow_explanation_prompt(context, flow_map)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по архитектуре ПО. Объясняй flow кода понятно и структурированно."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            explanation = response.choices[0].message.content
            
            return {
                "entry_function": entry_function,
                "flow_map": flow_map,
                "llm_explanation": explanation,
                "context": context
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM анализа flow: {e}[/red]")
            return {"error": f"Ошибка LLM анализа flow: {e}"}
    
    def answer_code_question(self, question: str, context_search: str = None) -> Dict[str, Any]:
        """Отвечает на вопросы о коде используя RAG"""
        
        # Если не указан контекст для поиска, используем сам вопрос
        search_query = context_search or question
        
        # Ищем релевантный код
        search_results = self.rag_engine.search(search_query, n_results=5)
        
        if not search_results:
            return {"error": "Не найден релевантный код для ответа на вопрос"}
        
        # Формируем промпт с контекстом
        prompt = self._create_question_answer_prompt(question, search_results)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт программист. Отвечай на вопросы о коде основываясь на предоставленном контексте. Будь точным и конкретным."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            return {
                "question": question,
                "answer": answer,
                "context_used": search_results,
                "search_query": search_query
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM ответа: {e}[/red]")
            return {"error": f"Ошибка LLM ответа: {e}"}
    
    def suggest_improvements(self, function_name: str) -> Dict[str, Any]:
        """Предлагает улучшения для функции"""
        
        context = self.rag_engine.get_function_context(function_name)
        
        if "error" in context:
            return context
        
        prompt = self._create_improvement_prompt(context)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты senior разработчик. Анализируй код и предлагай конкретные улучшения по производительности, читаемости и архитектуре."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            
            suggestions = response.choices[0].message.content
            
            return {
                "function_name": function_name,
                "context": context,
                "suggestions": suggestions,
                "complexity_score": context.get('flow_info', {}).get('complexity', 0)
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка LLM предложений: {e}[/red]")
            return {"error": f"Ошибка LLM предложений: {e}"}
    
    def find_similar_functions(self, function_name: str) -> Dict[str, Any]:
        """Находит похожие функции"""
        
        # Получаем контекст функции
        context = self.rag_engine.get_function_context(function_name)
        
        if "error" in context:
            return context
        
        # Извлекаем ключевые слова из функции для поиска
        main_function = context['main_function']
        function_code = main_function['document']
        
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
        
        # Анализируем сходства с помощью LLM
        prompt = self._create_similarity_analysis_prompt(main_function, similar_functions)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по анализу кода. Анализируй сходства между функциями и объясняй их назначение."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "target_function": function_name,
                "similar_functions": similar_functions,
                "similarity_analysis": analysis
            }
            
        except Exception as e:
            console.print(f"[red]Ошибка анализа сходства: {e}[/red]")
            return {
                "target_function": function_name,
                "similar_functions": similar_functions,
                "error": f"Ошибка LLM анализа: {e}"
            }
    
    def _create_function_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Создает промпт для анализа функции"""
        main_function = context['main_function']
        
        prompt_parts = [
            "Проанализируй следующую Python функцию:",
            f"\n**Функция:** {main_function['metadata']['name']}",
            f"**Файл:** {main_function['metadata']['file_path']}",
            f"**Строки:** {main_function['metadata']['line_start']}-{main_function['metadata']['line_end']}",
            f"**Сложность:** {main_function['metadata']['complexity']}",
            f"\n**Код:**\n{main_function['document']}",
        ]
        
        if context.get('callers'):
            callers = [c['metadata']['name'] for c in context['callers']]
            prompt_parts.append(f"\n**Вызывается из:** {', '.join(callers)}")
        
        if context.get('callees'):
            callees = [c['metadata']['name'] for c in context['callees']]
            prompt_parts.append(f"**Вызывает:** {', '.join(callees)}")
        
        prompt_parts.extend([
            "\nПроанализируй:",
            "1. Назначение и функциональность",
            "2. Качество кода и потенциальные проблемы",
            "3. Производительность и оптимизации",
            "4. Связи с другими функциями",
            "5. Рекомендации по улучшению"
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_flow_explanation_prompt(self, context: Dict[str, Any], flow_map: Dict[str, Any]) -> str:
        """Создает промпт для объяснения flow"""
        main_function = context['main_function']
        
        prompt_parts = [
            f"Объясни flow выполнения кода начиная с функции {main_function['metadata']['name']}:",
            f"\n**Основная функция:**\n{main_function['document']}",
            f"\n**Граф вызовов:**\n{json.dumps(flow_map, indent=2, ensure_ascii=False)}"
        ]
        
        if context.get('callees'):
            prompt_parts.append("\n**Вызываемые функции:**")
            for callee in context['callees'][:3]:
                prompt_parts.append(f"- {callee['metadata']['name']}: {callee['document'][:200]}...")
        
        prompt_parts.extend([
            "\nОбъясни:",
            "1. Последовательность выполнения",
            "2. Ключевые точки принятия решений",
            "3. Потоки данных между функциями",
            "4. Возможные пути выполнения",
            "5. Потенциальные точки отказа"
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_question_answer_prompt(self, question: str, search_results: List[Dict[str, Any]]) -> str:
        """Создает промпт для ответа на вопрос"""
        prompt_parts = [
            f"Вопрос: {question}",
            "\nКонтекст из кодовой базы:"
        ]
        
        for i, result in enumerate(search_results, 1):
            prompt_parts.append(f"\n--- Результат {i} ---")
            prompt_parts.append(f"Тип: {result['metadata']['type']}")
            prompt_parts.append(f"Имя: {result['metadata']['name']}")
            prompt_parts.append(f"Файл: {result['metadata']['file_path']}")
            prompt_parts.append(f"Код:\n{result['document'][:500]}...")
        
        prompt_parts.append("\nОтветь на вопрос основываясь на предоставленном контексте. Будь конкретным и приводи примеры из кода.")
        
        return "\n".join(prompt_parts)
    
    def _create_improvement_prompt(self, context: Dict[str, Any]) -> str:
        """Создает промпт для предложения улучшений"""
        main_function = context['main_function']
        
        prompt_parts = [
            f"Предложи улучшения для функции {main_function['metadata']['name']}:",
            f"\n**Код:**\n{main_function['document']}",
            f"\n**Сложность:** {main_function['metadata']['complexity']}",
            f"**Количество вызовов:** {main_function['metadata']['calls_count']}"
        ]
        
        if context.get('flow_info'):
            flow_info = context['flow_info']
            if flow_info.get('calls'):
                prompt_parts.append(f"**Вызывает функции:** {', '.join([c.split('::')[-1] for c in flow_info['calls']])}")
            if flow_info.get('called_by'):
                prompt_parts.append(f"**Вызывается из:** {', '.join([c.split('::')[-1] for c in flow_info['called_by']])}")
        
        prompt_parts.extend([
            "\nПредложи улучшения по:",
            "1. Производительности",
            "2. Читаемости кода",
            "3. Архитектуре и дизайну",
            "4. Обработке ошибок",
            "5. Тестируемости",
            "\nДай конкретные примеры кода для каждого предложения."
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_similarity_analysis_prompt(self, main_function: Dict[str, Any], similar_functions: List[Dict[str, Any]]) -> str:
        """Создает промпт для анализа сходства функций"""
        prompt_parts = [
            f"Проанализируй сходства между функцией {main_function['metadata']['name']} и другими функциями:",
            f"\n**Основная функция:**\n{main_function['document'][:300]}...",
            "\n**Похожие функции:**"
        ]
        
        for i, func in enumerate(similar_functions, 1):
            prompt_parts.append(f"\n{i}. {func['metadata']['name']} ({func['metadata']['file_path']}):")
            prompt_parts.append(f"{func['document'][:200]}...")
        
        prompt_parts.extend([
            "\nОпиши:",
            "1. Общие паттерны и подходы",
            "2. Различия в реализации",
            "3. Возможности для рефакторинга",
            "4. Общие утилитарные функции, которые можно выделить"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_flow_map(self, function_name: str, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Строит карту flow функций"""
        if current_depth >= max_depth:
            return {}
        
        context = self.rag_engine.get_function_context(function_name, include_callers=False, include_callees=True)
        
        if "error" in context:
            return {}
        
        flow_map = {
            "function": function_name,
            "calls": [],
            "depth": current_depth
        }
        
        if context.get('callees'):
            for callee in context['callees'][:3]:  # Ограничиваем количество
                callee_name = callee['metadata']['name']
                callee_flow = self._build_flow_map(callee_name, max_depth, current_depth + 1)
                if callee_flow:
                    flow_map["calls"].append(callee_flow)
                else:
                    flow_map["calls"].append({"function": callee_name, "calls": [], "depth": current_depth + 1})
        
        return flow_map
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Извлекает рекомендации из анализа"""
        # Простое извлечение рекомендаций по ключевым словам
        recommendations = []
        lines = analysis.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['рекомендую', 'предлагаю', 'стоит', 'можно улучшить', 'следует']):
                recommendations.append(line.strip())
        
        return recommendations[:5]  # Ограничиваем количество
    
    def display_analysis_result(self, result: Dict[str, Any], analysis_type: str = "function"):
        """Отображает результат анализа в красивом формате"""
        if "error" in result:
            console.print(Panel(f"[red]Ошибка: {result['error']}[/red]", title="Ошибка"))
            return
        
        if analysis_type == "function":
            self._display_function_analysis(result)
        elif analysis_type == "flow":
            self._display_flow_analysis(result)
        elif analysis_type == "question":
            self._display_question_answer(result)
        elif analysis_type == "improvements":
            self._display_improvements(result)
        elif analysis_type == "similar":
            self._display_similar_functions(result)
    
    def _display_function_analysis(self, result: Dict[str, Any]):
        """Отображает анализ функции"""
        function_name = result['function_name']
        analysis = result['llm_analysis']
        
        console.print(Panel(analysis, title=f"Анализ функции: {function_name}", border_style="blue"))
        
        if result.get('recommendations'):
            console.print("\n[bold yellow]Рекомендации:[/bold yellow]")
            for i, rec in enumerate(result['recommendations'], 1):
                console.print(f"{i}. {rec}")
    
    def _display_flow_analysis(self, result: Dict[str, Any]):
        """Отображает анализ flow"""
        entry_function = result['entry_function']
        explanation = result['llm_explanation']
        
        console.print(Panel(explanation, title=f"Flow анализ: {entry_function}", border_style="green"))
    
    def _display_question_answer(self, result: Dict[str, Any]):
        """Отображает ответ на вопрос"""
        question = result['question']
        answer = result['answer']
        
        console.print(Panel(f"[bold]Вопрос:[/bold] {question}\n\n[bold]Ответ:[/bold]\n{answer}", 
                          title="Ответ на вопрос", border_style="cyan"))
    
    def _display_improvements(self, result: Dict[str, Any]):
        """Отображает предложения по улучшению"""
        function_name = result['function_name']
        suggestions = result['suggestions']
        
        console.print(Panel(suggestions, title=f"Предложения по улучшению: {function_name}", border_style="magenta"))
    
    def _display_similar_functions(self, result: Dict[str, Any]):
        """Отображает анализ похожих функций"""
        target_function = result['target_function']
        analysis = result.get('similarity_analysis', 'Анализ недоступен')
        
        console.print(Panel(analysis, title=f"Похожие функции для: {target_function}", border_style="yellow")) 