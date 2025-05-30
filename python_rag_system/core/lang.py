import os

# Simple i18n dictionary for demo (extend as needed)
MESSAGES = {
    'ru': {
        'indexing_complete': '✅ Индексация завершена! Документация сохранена в {file}',
        'indexing_error': '❌ Ошибка индексации',
        'project_not_found': '❌ Директория проекта не найдена',
        'analysis_error': '❌ Ошибка анализа: {error}',
        'analysis_done': '🎉 Демонстрация завершена!',
        'goodbye': 'До свидания! 👋',
        'unknown_command': 'Неизвестная команда. Доступные: search, analyze, flow, ask, improve, similar, stats, quit',
        'prompt_command': 'Введите команду',
        'file_saved': '📄 Документация сохранена в {file}',
        'demo_interactive': 'Для интерактивного использования запустите:',
        'llm_unavailable': '⚠️  OPENAI_API_KEY не установлен. LLM функции недоступны.',
        'set_env': '   Установите переменную окружения для полной демонстрации:',
        'interrupted': 'Демонстрация прервана пользователем',
        'error': 'Ошибка: {error}',
    },
    'en': {
        'indexing_complete': '✅ Indexing complete! Documentation saved to {file}',
        'indexing_error': '❌ Indexing error',
        'project_not_found': '❌ Project directory not found',
        'analysis_error': '❌ Analysis error: {error}',
        'analysis_done': '🎉 Demo finished!',
        'goodbye': 'Goodbye! 👋',
        'unknown_command': 'Unknown command. Available: search, analyze, flow, ask, improve, similar, stats, quit',
        'prompt_command': 'Enter command',
        'file_saved': '📄 Documentation saved to {file}',
        'demo_interactive': 'For interactive usage, run:',
        'llm_unavailable': '⚠️  OPENAI_API_KEY is not set. LLM features unavailable.',
        'set_env': '   Set the environment variable for full demo:',
        'interrupted': 'Demo interrupted by user',
        'error': 'Error: {error}',
    }
}

def get_lang():
    return os.environ.get('RAG_LANG', 'ru').lower()

def tr(key, **kwargs):
    lang = get_lang()
    msg = MESSAGES.get(lang, MESSAGES['ru']).get(key, key)
    return msg.format(**kwargs)
