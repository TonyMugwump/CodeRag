import requests
import json
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()

class EmbeddingProvider(ABC):
    """Абстрактный базовый класс для провайдеров эмбеддингов"""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Кодирует тексты в эмбеддинги"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Возвращает размерность эмбеддингов"""
        pass

class SentenceTransformerProvider(EmbeddingProvider):
    """Провайдер для SentenceTransformers моделей"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        console.print(f"[green]Загружена SentenceTransformer модель: {model_name}[/green]")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class OllamaEmbeddingProvider(EmbeddingProvider):
    """Провайдер для Ollama эмбеддингов"""
    
    def __init__(self, 
                 model_name: str = "nomic-embed-text:v1.5",
                 base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embeddings"
        self._dimension = None
        
        # Проверяем доступность Ollama
        self._check_ollama_availability()
        console.print(f"[green]Подключен к Ollama модели: {model_name}[/green]")
    
    def _check_ollama_availability(self):
        """Проверяет доступность Ollama сервера"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name not in model_names:
                    console.print(f"[yellow]Предупреждение: модель {self.model_name} не найдена в Ollama[/yellow]")
                    console.print(f"[yellow]Доступные модели: {', '.join(model_names)}[/yellow]")
                    console.print(f"[yellow]Попробуйте: ollama pull {self.model_name}[/yellow]")
            else:
                raise Exception(f"Ollama API недоступен: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Не удается подключиться к Ollama: {e}")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    embeddings.append(embedding)
                else:
                    console.print(f"[red]Ошибка Ollama API: {response.status_code}[/red]")
                    console.print(f"[red]Ответ: {response.text}[/red]")
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Ошибка запроса к Ollama: {e}[/red]")
                raise
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Получает размерность эмбеддингов"""
        if self._dimension is None:
            # Тестовый запрос для определения размерности
            test_embedding = self.encode("test")
            self._dimension = len(test_embedding[0])
        
        return self._dimension

class EmbeddingFactory:
    """Фабрика для создания провайдеров эмбеддингов"""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> EmbeddingProvider:
        """Создает провайдер эмбеддингов"""
        
        if provider_type.lower() == "sentence_transformers":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            return SentenceTransformerProvider(model_name)
        
        elif provider_type.lower() == "ollama":
            model_name = kwargs.get("model_name", "nomic-embed-text:v1.5")
            base_url = kwargs.get("base_url", "http://localhost:11434")
            return OllamaEmbeddingProvider(model_name, base_url)
        
        else:
            raise ValueError(f"Неподдерживаемый тип провайдера: {provider_type}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Возвращает список доступных провайдеров"""
        return ["sentence_transformers", "ollama"] 