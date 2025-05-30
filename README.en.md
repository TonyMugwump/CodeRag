[Русская версия](README.md)

# 🐍 Python RAG System

A powerful code analysis system for Python and JavaScript (ES) using ChromaDB and LLM for advanced Retrieval-Augmented Generation (RAG).

## 🚀 Features

### 📊 Code Analysis
- **AST Parsing**: Deep analysis of Python and JavaScript code, extracting functions, classes, and methods
- **Call Graph**: Detailed function call graph construction
- **Dependency Graph**: Import and module dependency analysis
- **Complexity Metrics**: Automatic function complexity calculation

### 🔍 Search & Indexing
- **ChromaDB**: Vector database for semantic search
- **Embeddings**: Supports SentenceTransformers and Ollama (nomic-embed-text:v1.5)
- **Local Models**: Fully private processing via Ollama
- **Filtering**: Search by element type (functions, classes, imports)
- **Contextual Search**: Context-aware and relationship-based search

### 🧠 LLM Integration
- **Provider Support**: OpenAI GPT and local Ollama models
- **Function Analysis**: Detailed code analysis with LLM
- **Flow Explanations**: Understand program execution flow
- **Q&A**: Intelligent code questions and answers
- **Improvement Suggestions**: Optimization recommendations
- **Similar Function Search**: Pattern and similarity analysis
- **Local Processing**: Full privacy with Ollama

### 🛠️ Tools
- **CLI Interface**: Convenient command line
- **Interactive Mode**: Dialog interface for exploration
- **Documentation Export**: Automatic documentation generation
- **Mermaid Diagrams**: Call graph visualization

## 🟦 JavaScript (ES) Support

The system now supports analysis and indexing of JavaScript (ES) projects alongside Python!

- **JS Parsing**: Extracts functions, classes, imports from .js files
- **CLI**: Use `--language javascript` to index JS projects
- **Compatibility**: Works with modern ES code (no TypeScript)
- **Requirements**: Node.js and `esprima` package must be installed (`npm install esprima`)

### Example: Indexing a JS Project

```bash
python -m python_rag_system.cli index ./your_js_project --language javascript
```

### Example: Searching JS Code

```bash
python -m python_rag_system.cli search "function for data processing" --language javascript
```

### Programmatic Usage for JS

```python
from python_rag_system import PythonRAGEngine

rag_engine = PythonRAGEngine(
    collection_name="my_js_project",
    persist_directory=".chroma_js_store",
    language="javascript"
)
summary = rag_engine.index_project("./my_js_project")
print(f"Indexed {summary['total_elements']} JS elements")
```

### JavaScript (Node.js) Dependencies Setup

If you want to analyze JavaScript code, you need Node.js and npm.

1. Install Node.js: https://nodejs.org/
2. Initialize npm in the project root (if not already):
   ```sh
   npm init -y
   ```
3. Install required dependencies:
   ```sh
   npm install esprima
   ```
4. Now all JS analysis features are available!

> The script `js_parser.js` is used for batch parsing of JavaScript files.

### OpenAI API Setup (optional)
