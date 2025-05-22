#!/usr/bin/env python3
"""
Утилита для очистки коллекций ChromaDB
"""

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

console = Console()

def list_collections(persist_dir: str = ".chroma_rag_store"):
    """Показывает все коллекции"""
    try:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        
        if not collections:
            console.print("[yellow]Коллекции не найдены[/yellow]")
            return []
        
        table = Table(title="Коллекции ChromaDB")
        table.add_column("Имя", style="cyan")
        table.add_column("Документов", style="green")
        table.add_column("Метаданные", style="yellow")
        
        for collection in collections:
            count = collection.count()
            metadata = collection.metadata or {}
            metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
            
            table.add_row(collection.name, str(count), metadata_str)
        
        console.print(table)
        return collections
        
    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        return []

def cleanup_collections(persist_dir: str = ".chroma_rag_store"):
    """Очищает коллекции"""
    
    console.print("[bold blue]🧹 Утилита очистки коллекций ChromaDB[/bold blue]")
    
    collections = list_collections(persist_dir)
    
    if not collections:
        return
    
    console.print("\n[bold yellow]Опции очистки:[/bold yellow]")
    console.print("1. Удалить все коллекции")
    console.print("2. Удалить конкретную коллекцию")
    console.print("3. Удалить пустые коллекции")
    console.print("4. Выход")
    
    choice = console.input("\nВыберите опцию (1-4): ")
    
    try:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        if choice == "1":
            if Confirm.ask("Удалить ВСЕ коллекции? Это действие необратимо!"):
                for collection in collections:
                    client.delete_collection(collection.name)
                    console.print(f"[red]Удалена коллекция: {collection.name}[/red]")
                console.print("[green]Все коллекции удалены[/green]")
        
        elif choice == "2":
            collection_name = console.input("Введите имя коллекции для удаления: ")
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                if Confirm.ask(f"Удалить коллекцию '{collection_name}'?"):
                    client.delete_collection(collection_name)
                    console.print(f"[red]Удалена коллекция: {collection_name}[/red]")
            else:
                console.print(f"[yellow]Коллекция '{collection_name}' не найдена[/yellow]")
        
        elif choice == "3":
            empty_collections = [c for c in collections if c.count() == 0]
            
            if not empty_collections:
                console.print("[green]Пустых коллекций не найдено[/green]")
            else:
                console.print(f"Найдено {len(empty_collections)} пустых коллекций:")
                for collection in empty_collections:
                    console.print(f"  - {collection.name}")
                
                if Confirm.ask("Удалить все пустые коллекции?"):
                    for collection in empty_collections:
                        client.delete_collection(collection.name)
                        console.print(f"[red]Удалена пустая коллекция: {collection.name}[/red]")
        
        elif choice == "4":
            console.print("[green]Выход[/green]")
        
        else:
            console.print("[red]Неверный выбор[/red]")
            
    except Exception as e:
        console.print(f"[red]Ошибка при очистке: {e}[/red]")

def cleanup_all_stores():
    """Очищает все директории с коллекциями"""
    
    stores = [
        ".chroma_rag_store",
        ".chroma_ollama_store", 
        ".chroma_st_store",
        ".test_chroma_store",
        ".test_chroma_ollama_store"
    ]
    
    console.print("[bold blue]🧹 Очистка всех хранилищ ChromaDB[/bold blue]")
    
    for store in stores:
        console.print(f"\n[cyan]Проверяю {store}...[/cyan]")
        collections = list_collections(store)
        
        if collections:
            if Confirm.ask(f"Очистить {store}?"):
                cleanup_collections(store)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        cleanup_all_stores()
    else:
        cleanup_collections() 