def extract_text_from_file(meta: dict) -> str:
    if not isinstance(meta, dict) or 'filepath' not in meta or 'start_line' not in meta or 'end_line' not in meta:
        print(f"[extract] Некорректные метаданные: {meta}")
        return None
    try:
        with open(meta['filepath'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start = meta['start_line'] - 1
        end = meta['end_line']
        if start < 0 or end > len(lines) or start >= end:
            print(f"[extract] Некорректный диапазон строк: {start+1}-{end} для файла {meta['filepath']}")
            return None
        return ''.join(lines[start:end])
    except Exception as e:
        print(f"[extract] Ошибка при извлечении кода из {meta.get('filepath')}: {e}")
        return None