#!/usr/bin/env python3
"""
Пример использования Python RAG System с Ollama эмбеддингами
"""

import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from python_rag_system.core.rag_engine import PythonRAGEngine
from python_rag_system.core.llm_integration import LLMCodeAnalyzer

console = Console()

def main():
    console.print(Panel.fit("🐍 Демонстрация Python RAG System с Ollama эмбеддингами", style="bold blue"))
    
    # Конфигурация для Ollama
    embedding_config = {
        'embedding_provider': 'ollama',
        'embedding_model': 'nomic-embed-text:v1.5',
        'ollama_base_url': 'http://localhost:11434'
    }
    
    try:
        console.print("\n1. Инициализация RAG движка с Ollama...")
        
        # Создаем RAG движок с Ollama эмбеддингами
        rag_engine = PythonRAGEngine(
            collection_name="ollama_demo_collection_v2",  # Новое имя коллекции
            persist_directory=".chroma_ollama_store",
            **embedding_config
        )
        
        console.print("✅ RAG движок успешно инициализирован!")
        
        # Показываем статистику
        stats = rag_engine.get_collection_stats()
        
        table = Table(title="Конфигурация системы")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Провайдер эмбеддингов", stats['embedding_provider'])
        table.add_row("Модель эмбеддингов", stats['embedding_model'])
        table.add_row("Размерность эмбеддингов", str(stats['embedding_dimension']))
        table.add_row("Документов в коллекции", str(stats['total_documents']))
        
        console.print(table)
        
        # Если коллекция пустая, индексируем проект
        if stats['total_documents'] == 0:
            console.print("\n2. Индексация проекта...")
            project_path = "./python_rag_system"
            
            summary = rag_engine.index_project(project_path)
            
            if summary:
                console.print(f"✅ Проиндексировано {summary['total_elements']} элементов")
            else:
                console.print("❌ Ошибка индексации")
                return
        else:
            console.print("\n2. Используем существующий индекс...")
        
        # Демонстрация поиска
        console.print("\n3. Демонстрация семантического поиска...")
        
        search_queries = [
            "функция для парсинга Python кода",
            "создание эмбеддингов",
            "работа с ChromaDB",
            "анализ сложности кода"
        ]
        
        for query in search_queries:
            console.print(f"\n🔍 Поиск: '{query}'")
            
            results = rag_engine.search(query, n_results=3)
            
            if results:
                for i, result in enumerate(results[:2], 1):
                    metadata = result['metadata']
                    distance = result.get('distance', 0)
                    relevance = 1 - distance
                    
                    console.print(f"  {i}. {metadata['name']} ({metadata['type']}) - релевантность: {relevance:.3f}")
                    console.print(f"     📁 {metadata['file_path']}:{metadata['line_start']}")
            else:
                console.print("  Результаты не найдены")
        
        # Демонстрация анализа функций
        console.print("\n4. Демонстрация анализа функций...")
        
        # Ищем интересную функцию для анализа
        function_results = rag_engine.search("parse_file", filter_type="function", n_results=1)
        
        if function_results:
            function_name = function_results[0]['metadata']['name']
            console.print(f"\n🧠 Анализ функции: {function_name}")
            
            # Получаем контекст функции
            context = rag_engine.get_function_context(function_name)
            
            if 'error' not in context:
                console.print(f"  📊 Сложность: {context['main_function']['metadata']['complexity']}")
                
                if context.get('flow_info'):
                    flow = context['flow_info']
                    if flow.get('calls'):
                        console.print(f"  📞 Вызывает: {len(flow['calls'])} функций")
                    if flow.get('called_by'):
                        console.print(f"  📲 Вызывается из: {len(flow['called_by'])} мест")
        
        # Демонстрация сравнения с SentenceTransformers
        console.print("\n5. Сравнение с SentenceTransformers...")
        
        # Создаем второй движок с SentenceTransformers для сравнения
        st_engine = PythonRAGEngine(
            collection_name="st_demo_collection",
            persist_directory=".chroma_st_store",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Сравниваем результаты поиска
        test_query = "функция для создания эмбеддингов"
        
        console.print(f"\n🔍 Сравнение результатов для запроса: '{test_query}'")
        
        ollama_results = rag_engine.search(test_query, n_results=3)
        st_results = st_engine.search(test_query, n_results=3)
        
        comparison_table = Table(title="Сравнение результатов поиска")
        comparison_table.add_column("Позиция", style="cyan")
        comparison_table.add_column("Ollama (nomic-embed-text)", style="green")
        comparison_table.add_column("SentenceTransformers (MiniLM)", style="yellow")
        
        max_results = max(len(ollama_results), len(st_results))
        
        for i in range(min(3, max_results)):
            ollama_name = ollama_results[i]['metadata']['name'] if i < len(ollama_results) else "N/A"
            st_name = st_results[i]['metadata']['name'] if i < len(st_results) else "N/A"
            
            ollama_rel = f"{1-ollama_results[i].get('distance', 1):.3f}" if i < len(ollama_results) else "N/A"
            st_rel = f"{1-st_results[i].get('distance', 1):.3f}" if i < len(st_results) else "N/A"
            
            comparison_table.add_row(
                str(i+1),
                f"{ollama_name} ({ollama_rel})",
                f"{st_name} ({st_rel})"
            )
        
        console.print(comparison_table)
        
        console.print("\n✅ Демонстрация завершена!")
        
        # Рекомендации
        recommendations = Panel(
            """
[bold green]Рекомендации по использованию Ollama эмбеддингов:[/bold green]

• [bold]nomic-embed-text:v1.5[/bold] показывает отличные результаты для кода
• Поддерживает контекст до 8192 токенов (vs 512 у MiniLM)
• Полностью локальная обработка - данные не покидают ваш сервер
• Для установки: [cyan]ollama pull nomic-embed-text:v1.5[/cyan]
• Убедитесь что Ollama запущен: [cyan]ollama serve[/cyan]

[bold yellow]Альтернативные модели для экспериментов:[/bold yellow]
• [cyan]mxbai-embed-large[/cyan] - большая модель с высокой точностью
• [cyan]snowflake-arctic-embed[/cyan] - специализированная для поиска
            """,
            title="💡 Советы",
            border_style="blue"
        )
        
        console.print(recommendations)
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка: {e}[/red]")
        console.print("\n[yellow]Убедитесь что:[/yellow]")
        console.print("1. Ollama запущен: ollama serve")
        console.print("2. Модель установлена: ollama pull nomic-embed-text:v1.5")
        console.print("3. Сервер доступен по адресу http://localhost:11434")

if __name__ == "__main__":
    main() 