# Project Flow RAG: Семантическая карта архитектуры и сценариев

---

## 1. Цель Project Flow RAG

Project Flow RAG — это не просто индекс функций, а семантическая карта архитектуры проекта:
- Связи между модулями, функциями, классами, точками входа, сценариями использования.
- Основные flows (пайплайны, сценарии, бизнес-логика).
- Сущности и данные, которые проходят через проект.
- High-level workflow (например, "запрос пользователя → обработка → ответ").

---

## 2. Для чего нужен этот RAG

- **Вася**:
  - Понимает, о чём спрашивает пользователь ("как работает авторизация?", "где точка входа?", "какой pipeline для загрузки данных?").
  - Может объяснить, нарисовать flow, подсказать, где что реализовано.
  - Может уточнить, какой компонент/flow пользователь хочет изменить.
- **Михаил**:
  - Использует этот RAG для архитектурных решений: "где лучше внедрить новую фичу?", "какие функции участвуют в этом сценарии?", "что затронет изменение?".
  - Может строить dependency resolution, impact analysis, traceability.

---

## 3. Структура Project Flow RAG

### 3.1. Узлы (Nodes)
- **Function**: отдельная функция (имя, файл, docstring, координаты)
- **Class**: класс (имя, файл, docstring)
- **Module**: модуль/файл
- **Flow/Scenario**: сценарий, pipeline, бизнес-логика (может быть извлечён из docstring, комментариев, sequence diagrams)
- **EntryPoint**: точка входа (например, main, обработчик запроса)
- **DataEntity**: сущность данных (например, User, Order, DataChunk)

### 3.2. Связи (Edges)
- **calls**: функция вызывает функцию
- **implements**: функция/класс реализует flow/scenario
- **entry_for**: функция/модуль является точкой входа для flow
- **uses**: функция/класс использует сущность данных
- **depends_on**: flow зависит от другого flow/модуля
- **part_of**: функция/класс входит в flow

---

## 4. Пример структуры данных (Python)

```python
class FlowRAG:
    def __init__(self):
        self.nodes = {}  # node_id -> node_info
        self.edges = []  # (from_id, to_id, relation)

    def add_node(self, node_id, node_type, **attrs):
        self.nodes[node_id] = {"type": node_type, **attrs}

    def add_edge(self, from_id, to_id, relation):
        self.edges.append((from_id, to_id, relation))
```

### Пример наполнения

```python
rag = FlowRAG()
# Узлы
rag.add_node("auth.py:login", "Function", doc="User login", file="auth.py")
rag.add_node("UserAuthenticationFlow", "Flow", description="User login and session management")
rag.add_node("main.py:main", "EntryPoint", doc="Main entry point", file="main.py")
# Связи
rag.add_edge("main.py:main", "UserAuthenticationFlow", "entry_for")
rag.add_edge("UserAuthenticationFlow", "auth.py:login", "implements")
rag.add_edge("auth.py:login", "db.py:query", "calls")
```

---

## 5. Как строить такой RAG

### 5.1. Автоматически
- **AST-парсинг**: извлекаем функции, классы, связи вызовов.
- **Docstring/комментарии**: ищем ключевые слова ("flow", "pipeline", "scenario", "entry point").
- **Sequence diagrams/mermaid**: если есть, парсим и добавляем как flows.
- **LLM-анализ**: если docstring слабые, просим LLM сгенерировать summary, выделить сценарии.

### 5.2. Пример кода для автоматического сбора flows

```python
def extract_flows_from_docstring(docstring):
    flows = []
    if docstring:
        for line in docstring.splitlines():
            if "flow" in line.lower() or "pipeline" in line.lower() or "scenario" in line.lower():
                flows.append(line.strip())
    return flows
```

---

## 6. Пример пайплайна построения Project Flow RAG

```python
from project_map import collect_functions_and_calls

def build_project_flow_rag(project_path):
    function_map, call_graph = collect_functions_and_calls(project_path)
    rag = FlowRAG()
    # 1. Добавляем функции и классы как узлы
    for func_id, info in function_map.items():
        rag.add_node(func_id, "Function", doc=info["doc"], file=info["file"])
        # 2. Извлекаем flows из docstring
        flows = extract_flows_from_docstring(info["doc"])
        for flow in flows:
            flow_id = flow.replace(" ", "_")
            rag.add_node(flow_id, "Flow", description=flow)
            rag.add_edge(flow_id, func_id, "implements")
    # 3. Добавляем связи вызовов
    for caller, callees in call_graph.items():
        for callee in callees:
            rag.add_edge(caller, callee, "calls")
    # 4. (Опционально) ищем точки входа
    # ... (например, main.py:main, app.py:run, etc.)
    return rag
```

---

## 7. Использование

- **Вася**:
  - Поиск по flows, сценариям, точкам входа, объяснение архитектуры.
  - Ответы на вопросы "как работает X?" "где реализован pipeline Y?"
- **Михаил**:
  - Для архитектурных решений, поиска точек внедрения, анализа влияния изменений.

---

## 8. Визуализация

- Можно экспортировать rag.edges в mermaid-graph или graphviz для визуального анализа.

---

## 9. Итог

- Project Flow RAG — это граф, где узлы — не только функции, но и сценарии, точки входа, сущности, а рёбра — семантические связи.
- Такой RAG позволяет Ваcе и Михаилу работать на уровне архитектуры и сценариев, а не только функций.

---

**Если нужно реализовать прототип такого пайплайна — дай знать!** 