#!/usr/bin/env python3
"""
Скрипт для проверки и настройки Ollama для Python RAG System
"""

import requests
import subprocess
import sys
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_ollama_server(base_url="http://localhost:11434"):
    """Проверяет доступность Ollama сервера"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_installed_models(base_url="http://localhost:11434"):
    """Получает список установленных моделей"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except:
        return []

def pull_model(model_name):
    """Устанавливает модель через ollama pull"""
    try:
        console.print(f"[yellow]Установка модели {model_name}...[/yellow]")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 минут таймаут
        )
        
        if result.returncode == 0:
            console.print(f"[green]✅ Модель {model_name} установлена успешно[/green]")
            return True
        else:
            console.print(f"[red]❌ Ошибка установки {model_name}: {result.stderr}[/red]")
            return False
    except subprocess.TimeoutExpired:
        console.print(f"[red]❌ Таймаут при установке {model_name}[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]❌ Ollama CLI не найден. Установите Ollama: https://ollama.ai[/red]")
        return False

def start_ollama_server():
    """Пытается запустить Ollama сервер"""
    try:
        console.print("[yellow]Попытка запуска Ollama сервера...[/yellow]")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Ждем запуска
        for i in range(10):
            time.sleep(1)
            if check_ollama_server():
                console.print("[green]✅ Ollama сервер запущен[/green]")
                return True
        
        console.print("[red]❌ Не удалось запустить Ollama сервер[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]❌ Ollama CLI не найден. Установите Ollama: https://ollama.ai[/red]")
        return False

def main():
    console.print(Panel(
        "[bold blue]🦙 Настройка Ollama для Python RAG System[/bold blue]",
        border_style="blue"
    ))
    
    # Рекомендуемые модели
    recommended_models = {
        "nomic-embed-text:v1.5": "Эмбеддинги (768 dim, 8192 tokens)",
        "gemma3:27b": "LLM для анализа кода (27B параметров)",
        "gemma3:12b": "LLM альтернатива (9B параметров, быстрее)",
    }
    
    # 1. Проверяем Ollama сервер
    console.print("\n[yellow]1. Проверка Ollama сервера...[/yellow]")
    
    if not check_ollama_server():
        console.print("[red]❌ Ollama сервер недоступен[/red]")
        
        # Пытаемся запустить
        if not start_ollama_server():
            console.print("\n[red]Для запуска Ollama выполните:[/red]")
            console.print("[red]  ollama serve[/red]")
            console.print("\n[red]Если Ollama не установлен:[/red]")
            console.print("[red]  curl -fsSL https://ollama.ai/install.sh | sh[/red]")
            return
    else:
        console.print("[green]✅ Ollama сервер доступен[/green]")
    
    # 2. Проверяем установленные модели
    console.print("\n[yellow]2. Проверка установленных моделей...[/yellow]")
    installed_models = get_installed_models()
    
    if installed_models:
        table = Table(title="Установленные модели")
        table.add_column("Модель", style="cyan")
        table.add_column("Статус", style="green")
        
        for model in installed_models:
            table.add_row(model, "✅ Установлена")
        
        console.print(table)
    else:
        console.print("[yellow]Модели не найдены[/yellow]")
    
    # 3. Рекомендации по установке
    console.print("\n[yellow]3. Рекомендуемые модели для RAG системы:[/yellow]")
    
    rec_table = Table(title="Рекомендуемые модели")
    rec_table.add_column("Модель", style="cyan")
    rec_table.add_column("Описание", style="white")
    rec_table.add_column("Статус", style="green")
    
    for model, description in recommended_models.items():
        status = "✅ Установлена" if model in installed_models else "❌ Не установлена"
        rec_table.add_row(model, description, status)
    
    console.print(rec_table)
    
    # 4. Предлагаем установить недостающие модели
    missing_models = [model for model in recommended_models.keys() if model not in installed_models]
    
    if missing_models:
        console.print(f"\n[yellow]Найдено {len(missing_models)} недостающих моделей[/yellow]")
        
        for model in missing_models:
            if console.input(f"Установить {model}? [y/N]: ").lower().startswith('y'):
                pull_model(model)
    
    # 5. Финальная проверка
    console.print("\n[yellow]5. Финальная проверка...[/yellow]")
    final_models = get_installed_models()
    
    has_embedding = any("nomic-embed-text" in model for model in final_models)
    has_llm = any(model.startswith(("gemma2", "llama3")) for model in final_models)
    
    if has_embedding and has_llm:
        console.print("[green]🎉 Ollama настроен успешно![/green]")
        console.print("\n[blue]Теперь вы можете использовать:[/blue]")
        console.print("[blue]python example_ollama_llm_usage.py[/blue]")
        console.print("[blue]python -m python_rag_system.cli interactive --llm-provider ollama[/blue]")
    else:
        console.print("[yellow]⚠️  Рекомендуется установить как минимум:[/yellow]")
        if not has_embedding:
            console.print("[yellow]  - nomic-embed-text:v1.5 (для эмбеддингов)[/yellow]")
        if not has_llm:
            console.print("[yellow]  - gemma2:27b или gemma2:9b (для LLM анализа)[/yellow]")
    
    # 6. Информация о размерах моделей
    console.print("\n[cyan]💡 Информация о размерах моделей:[/cyan]")
    console.print("[cyan]  - nomic-embed-text:v1.5: ~274MB[/cyan]")
    console.print("[cyan]  - gemma2:9b: ~5.4GB[/cyan]")
    console.print("[cyan]  - gemma2:27b: ~15GB[/cyan]")
    console.print("[cyan]  - llama3.1:8b: ~4.7GB[/cyan]")

if __name__ == "__main__":
    main() 