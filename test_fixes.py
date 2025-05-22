#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений в RAG системе
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python_rag_system'))

from python_rag_system.core.rag_engine import PythonRAGEngine
from rich.console import Console
from rich.table import Table

console = Console()

def test_function_search():
    """Тестирует поиск функций"""
    console.print("\n[bold blue]🔍 Тестирование поиска функций[/bold blue]")
    
    # Инициализируем RAG движок
    rag = PythonRAGEngine(
        collection_name="test_rag",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Получаем список доступных функций
    console.print("\n[yellow]Получение списка доступных функций...[/yellow]")
    functions = rag.list_available_functions()
    
    if functions:
        console.print(f"[green]Найдено {len(functions)} функций[/green]")
        
        # Показываем первые 10 функций
        table = Table(title="Доступные функции")
        table.add_column("№", style="cyan")
        table.add_column("Имя функции", style="green")
        
        for i, func_name in enumerate(functions[:10], 1):
            table.add_row(str(i), func_name)
        
        console.print(table)
        
        # Тестируем поиск конкретной функции
        if functions:
            test_function = functions[0]
            console.print(f"\n[yellow]Тестирование поиска функции: {test_function}[/yellow]")
            
            context = rag.get_function_context(test_function)
            
            if "error" in context:
                console.print(f"[red]Ошибка: {context['error']}[/red]")
            else:
                console.print(f"[green]✅ Функция найдена успешно![/green]")
                console.print(f"Тип: {context['main_function']['metadata']['type']}")
                console.print(f"Файл: {context['main_function']['metadata']['file_path']}")
                console.print(f"Строки: {context['main_function']['metadata']['line_start']}-{context['main_function']['metadata']['line_end']}")
        
        # Тестируем поиск несуществующей функции
        console.print(f"\n[yellow]Тестирование поиска несуществующей функции...[/yellow]")
        context = rag.get_function_context("_nonexistent_function_12345")
        
        if "error" in context:
            console.print(f"[green]✅ Корректно обработана ошибка: {context['error'][:100]}...[/green]")
        else:
            console.print(f"[red]❌ Ожидалась ошибка, но функция найдена[/red]")
    
    else:
        console.print("[red]❌ Функции не найдены. Возможно, проект не проиндексирован.[/red]")

def test_class_search():
    """Тестирует поиск классов"""
    console.print("\n[bold blue]🔍 Тестирование поиска классов[/bold blue]")
    
    rag = PythonRAGEngine(
        collection_name="test_rag",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Получаем список доступных классов
    console.print("\n[yellow]Получение списка доступных классов...[/yellow]")
    classes = rag.list_available_classes()
    
    if classes:
        console.print(f"[green]Найдено {len(classes)} классов[/green]")
        
        # Показываем классы
        table = Table(title="Доступные классы")
        table.add_column("№", style="cyan")
        table.add_column("Имя класса", style="green")
        
        for i, class_name in enumerate(classes[:10], 1):
            table.add_row(str(i), class_name)
        
        console.print(table)
        
        # Тестируем поиск конкретного класса
        if classes:
            test_class = classes[0]
            console.print(f"\n[yellow]Тестирование поиска класса: {test_class}[/yellow]")
            
            context = rag.get_class_context(test_class)
            
            if "error" in context:
                console.print(f"[red]Ошибка: {context['error']}[/red]")
            else:
                console.print(f"[green]✅ Класс найден успешно![/green]")
                console.print(f"Файл: {context['main_class']['metadata']['file_path']}")
                console.print(f"Методов найдено: {len(context['methods'])}")
    
    else:
        console.print("[yellow]Классы не найдены в проекте[/yellow]")

def test_search_by_name():
    """Тестирует новый метод search_by_name"""
    console.print("\n[bold blue]🔍 Тестирование метода search_by_name[/bold blue]")
    
    rag = PythonRAGEngine(
        collection_name="test_rag",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Получаем статистику коллекции
    stats = rag.get_collection_stats()
    console.print(f"\n[cyan]Статистика коллекции:[/cyan]")
    console.print(f"Всего документов: {stats['total_documents']}")
    console.print(f"Провайдер эмбеддингов: {stats['embedding_provider']}")
    console.print(f"Модель: {stats['embedding_model']}")
    console.print(f"Размерность: {stats['embedding_dimension']}")
    
    if stats['total_documents'] > 0:
        # Тестируем поиск по имени
        console.print(f"\n[yellow]Тестирование search_by_name...[/yellow]")
        
        # Пробуем найти любую функцию
        results = rag.search_by_name("__init__", "function")
        if results:
            console.print(f"[green]✅ Найдено {len(results)} результатов для '__init__'[/green]")
        else:
            console.print(f"[yellow]Функция '__init__' не найдена[/yellow]")
        
        # Пробуем найти любой класс
        results = rag.search_by_name("PythonRAGEngine", "class")
        if results:
            console.print(f"[green]✅ Найдено {len(results)} результатов для 'PythonRAGEngine'[/green]")
        else:
            console.print(f"[yellow]Класс 'PythonRAGEngine' не найден[/yellow]")
    
    else:
        console.print("[red]❌ Коллекция пуста. Необходимо проиндексировать проект.[/red]")

def main():
    """Основная функция тестирования"""
    console.print("[bold green]🚀 Запуск тестов исправлений RAG системы[/bold green]")
    
    try:
        test_search_by_name()
        test_function_search()
        test_class_search()
        
        console.print("\n[bold green]✅ Все тесты завершены![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Ошибка во время тестирования: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

if __name__ == "__main__":
    main() 