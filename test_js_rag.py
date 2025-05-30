import os
import tempfile
import shutil
import pytest
from python_rag_system.core.rag_engine import PythonRAGEngine

JS_TEST_CODE = '''
function foo(a, b) {
    return a + b;
}

class Bar {
    method(x) {
        return foo(x, 2);
    }
}
import baz from "baz";
'''

def test_js_indexing_and_search(tmp_path):
    # Create a temporary JS file
    js_file = tmp_path / "test.js"
    js_file.write_text(JS_TEST_CODE)
    # Create a new collection directory
    persist_dir = tmp_path / ".chroma_rag_store"
    # Index the JS file
    engine = PythonRAGEngine(
        collection_name="test_js_rag",
        persist_directory=str(persist_dir),
        language="javascript"
    )
    summary = engine.index_project(str(tmp_path))
    assert summary["functions"] >= 1
    assert summary["classes"] >= 1
    # Test search for function
    results = engine.search_by_name("foo", element_type="function")
    assert any("foo" in r["metadata"]["name"] for r in results)
    # Test search for class
    class_results = engine.search_by_name("Bar", element_type="class")
    assert any("Bar" in r["metadata"]["name"] for r in class_results)
    # Test file summary
    file_summary = engine.get_file_summary(str(js_file))
    assert file_summary["total_elements"] > 0

def test_js_indexing_no_crash(tmp_path):
    # Should not crash on empty JS file
    js_file = tmp_path / "empty.js"
    js_file.write_text("")
    persist_dir = tmp_path / ".chroma_rag_store"
    engine = PythonRAGEngine(
        collection_name="test_js_empty",
        persist_directory=str(persist_dir),
        language="javascript"
    )
    summary = engine.index_project(str(tmp_path))
    assert isinstance(summary, dict)
