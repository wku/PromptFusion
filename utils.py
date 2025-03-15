"""
Модуль утилит для CodebaseGPT

Объединяет вспомогательные функции из разных модулей:
- utils.py
- init_utils.py
- input_utils.py
- code_utils.py
- token_utils.py
- cost_utils.py
- pretty_bytes.py
- is_text_or_bin.py
"""

import os
import re
import json
import logging
import pathspec
import chardet
from typing import List, Dict, Optional, Any, Tuple, Set, Union
from wcmatch import glob
import tiktoken
from config import DATA_ROOT, LARGE_SOURCE_FILE
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Будем импортировать модели только если они нам нужны
# Это помогает избежать циклических импортов
_FileNode = None
_ProjStat = None
_FileState = None
_ProjState = None


def _lazy_import_models():
    """Ленивая загрузка моделей для избежания циклических импортов"""
    global _FileNode, _ProjStat, _FileState, _ProjState
    if _FileNode is None:
        from models import FileNode, ProjStat, FileState, ProjState
        _FileNode = FileNode
        _ProjStat = ProjStat
        _FileState = FileState
        _ProjState = ProjState


# Настройка логирования
def setup_logging() -> logging.Logger:
    """Настраивает и возвращает логгер приложения"""
    logger = logging.getLogger ('codebasegpt')
    logger.setLevel (logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler ()
        formatter = logging.Formatter ('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter (formatter)
        logger.addHandler (handler)

        # Добавим файловый обработчик
        if not os.path.exists ('logs'):
            os.makedirs ('logs')
        file_handler = logging.FileHandler ('logs/codebasegpt.log')
        file_handler.setFormatter (formatter)
        logger.addHandler (file_handler)

    return logger


# Утилиты для работы с вводом пользователя
def input_yes_no(prompt: str) -> str:
    """Запрашивает у пользователя ответ Да/Нет и возвращает его ввод"""
    while True:
        user_input = input (prompt)
        if is_yes (user_input) or is_no (user_input) or is_default (user_input):
            return user_input
        else:
            print ("Ошибка: Введите 'y', 'n' или оставьте пустым для значения по умолчанию.")


def is_yes(input_str: str) -> bool:
    """Проверяет, означает ли ввод пользователя 'Да'"""
    trimmed_input = input_str.strip ().lower ()
    return trimmed_input in ['y', 'yes', 'да', 'д', 'ok', 'ок']


def is_no(input_str: str) -> bool:
    """Проверяет, означает ли ввод пользователя 'Нет'"""
    trimmed_input = input_str.strip ().lower ()
    return trimmed_input in ['n', 'no', 'нет', 'н']


def is_default(input_str: str) -> bool:
    """Проверяет, что пользователь ввел пустую строку (значение по умолчанию)"""
    return input_str.strip () == ''


# Утилиты для форматирования вывода
def bytes_to_str(num: int, suffix: str = 'B') -> str:
    """Преобразует количество байтов в удобочитаемую строку"""
    k_base = 1000.0
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs (num) < k_base:
            if unit == '':
                return f"{num:.0f} {unit}{suffix}"
            else:
                return f"{num:3.1f} {unit}{suffix}"
        num /= k_base
    return f"{num:.1f} Y{suffix}"


# Утилиты для работы с файлами
def is_folder_exist(path_to_check: str) -> bool:
    """Проверяет, существует ли папка по указанному пути"""
    if os.path.exists (path_to_check):
        stat = os.stat (path_to_check)
        return stat.st_mode & 0o170000 == 0o040000
    return False


def load_gitignore(root_path: str) -> pathspec.PathSpec:
    """Загружает правила .gitignore из корня проекта"""
    gitignore_file = os.path.join (root_path, '.gitignore')
    if os.path.exists (gitignore_file):
        with open (gitignore_file, 'r', encoding='utf-8') as f:
            return pathspec.PathSpec.from_lines ('gitwildmatch', f.read ().splitlines ())
    else:
        print ('Файл .gitignore не найден')
        return pathspec.PathSpec.from_lines ('gitwildmatch', [])


def is_text_file(file_path: str) -> bool:
    """Определяет, является ли файл текстовым или двоичным"""
    # Проверка по расширению
    if is_text_by_ext (file_path):
        return True
    if is_bin_by_ext (file_path):
        return False

    # Проверка по содержимому
    return is_text_by_enc (file_path)


def is_text_by_ext(file_path: str) -> bool:
    """Проверяет, является ли файл текстовым по его расширению"""
    # Список текстовых расширений
    text_exts = [
        'py', 'js', 'jsx', 'ts', 'tsx', 'html', 'css', 'md', 'txt', 'json', 'yaml', 'yml',
        'c', 'cpp', 'h', 'hpp', 'java', 'cs', 'go', 'rs', 'php', 'rb', 'sh', 'bash', 'bat',
        'xml', 'conf', 'ini', 'sql', 'csv', 'toml'
    ]
    ext = os.path.splitext (file_path)[1][1:].lower ()
    return ext in text_exts


def is_bin_by_ext(file_path: str) -> bool:
    """Проверяет, является ли файл двоичным по его расширению"""
    # Список двоичных расширений
    binary_exts = [
        'exe', 'dll', 'so', 'dylib', 'bin', 'obj', 'o', 'a', 'lib', 'pyd', 'pyc',
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'ico', 'pdf', 'doc', 'docx',
        'xls', 'xlsx', 'ppt', 'pptx', 'zip', 'tar', 'gz', 'rar', '7z', 'db', 'sqlite'
    ]
    ext = os.path.splitext (file_path)[1][1:].lower ()
    return ext in binary_exts


def is_text_by_enc(file_path: str) -> bool:
    """Проверяет, является ли файл текстовым по его кодировке"""
    try:
        with open (file_path, 'rb') as file:
            raw_data = file.read (512)
            encoding = chardet.detect (raw_data)['encoding']
            try:
                if encoding:
                    raw_data.decode (encoding)
                return True
            except (UnicodeDecodeError, TypeError):
                return False
    except:
        return False


# Утилиты для работы с кодом
def remove_comments(file_name: str, file_content: str) -> str:
    """Удаляет комментарии из кода в зависимости от типа файла"""
    file_extension = file_name.split ('.')[-1].lower () if '.' in file_name else ''

    if file_extension in ['js', 'jsx', 'mjs', 'cjs', 'es', 'es6', 'ts', 'tsx', 'mts', 'java', 'c', 'h', 'cpp', 'cs']:
        # Удаляем блочные комментарии
        file_content = re.sub (r'/\*[\s\S]*?\*/', '', file_content)

        # Удаляем строчные комментарии
        lines = file_content.splitlines ()
        lines = [line for line in lines if not line.strip ().startswith ('//')]
        lines = [line.split ('//')[0].rstrip () for line in lines]
        return '\n'.join (lines)

    elif file_extension == 'css':
        return re.sub (r'/\*[\s\S]*?\*/', '', file_content)

    elif file_extension == 'html':
        return re.sub (r'<!--[^>]*-->', '', file_content)

    elif file_extension == 'py':
        lines = file_content.splitlines ()
        lines = [line for line in lines if not line.strip ().startswith ('#')]
        lines = [line.split ('#')[0].rstrip () for line in lines]
        return '\n'.join (lines)

    else:
        return file_content


def trim_code(text: str) -> str:
    """Удаляет лишние пустые строки из кода"""
    lines = text.splitlines ()
    result = []

    # Заменяем более 1 пустой строки просто на 1 пустую строку
    for line in lines:
        if line.strip () == '' and (not result or result[-1].strip () == ''):
            continue
        result.append (line)

    # Удаляем пустые строки в конце файла
    while result and result[-1].strip () == '':
        result.pop ()

    return '\n'.join (result)


# Утилиты для токенизации и оценки стоимости
def get_tokens_cnt(text: str) -> int:
    """Подсчитывает количество токенов в тексте"""
    encoder = tiktoken.encoding_for_model ('gpt-3.5-turbo')
    return len (encoder.encode (text, disallowed_special=()))


def limit_string(text: str, max_tokens: int) -> str:
    """Ограничивает длину строки до указанного количества токенов"""
    encoder = tiktoken.encoding_for_model ('gpt-3.5-turbo')
    tokens = encoder.encode (text, disallowed_special=())
    if len (tokens) > max_tokens:
        return encoder.decode (tokens[:max_tokens])
    return text


# Ценовые модели для разных API
model_pricing = [
    {
        'model': 'qwen/qwen-turbo',
        'input_cost': 0.00005,  # $0.05/M = $0.00005/токен
        'output_cost': 0.0002,  # $0.2/M = $0.0002/токен
        'image_cost': None,
        'request_cost': None
    },
    {
        'model': 'openai/gpt-4o-2024-11-20',
        'input_cost': 0.0025,  # $2.5/M = $0.0025/токен
        'output_cost': 0.01,  # $10/M = $0.01/токен
        'image_cost': 0.003613,  # $3.613/K = $0.003613/KB
        'request_cost': None
    },
    {
        'model': 'openai/gpt-4o-mini-search-preview',
        'input_cost': 0.00015,  # $0.15/M = $0.00015/токен
        'output_cost': 0.0006,  # $0.6/M = $0.0006/токен
        'image_cost': 0.000217,  # $0.217/K = $0.000217/KB
        'request_cost': 0.0275  # $27.5/K = $0.0275/запрос
    },
    {
        'model': 'openai/gpt-4o-mini',
        'input_cost': 0.00015,  # $0.15/M = $0.00015/токен
        'output_cost': 0.0006,  # $0.6/M = $0.0006/токен
        'image_cost': 0.000217,  # $0.217/K = $0.000217/KB
        'request_cost': None
    },
    {
        'model': 'openai/gpt-4o',
        'input_cost': 0.005,
        'output_cost': 0.015,
        'image_cost': None,
        'request_cost': None
    },
    {
        'model': 'text-embedding-ada-002',
        'input_cost': 0.0,
        'output_cost': 0.0,  # Embedding не генерирует выходные токены
        'image_cost': None,
        'request_cost': None
    }
]

def get_cost(model: str, in_tokens: int, out_tokens: int, image_size_kb: int = 0) -> float:
    """
    Рассчитывает стоимость запроса на основе модели, количества токенов и размера изображений

    Args:
        model: идентификатор модели
        in_tokens: количество входных токенов
        out_tokens: количество выходных токенов
        image_size_kb: размер изображений в KB (для моделей с поддержкой изображений)

    Returns:
        Общая стоимость запроса в долларах США
    """
    pricing = next ((p for p in model_pricing if model.endswith (p['model'])), None)
    if not pricing:
        return 0.0

    cost = 0.0

    # Учитываем стоимость входных токенов
    if pricing['input_cost']:
        cost += pricing['input_cost'] * (in_tokens / 1000)

    # Учитываем стоимость выходных токенов
    if pricing['output_cost']:
        cost += pricing['output_cost'] * (out_tokens / 1000)

    # Учитываем стоимость изображений, если они поддерживаются
    if pricing['image_cost'] and image_size_kb > 0:
        cost += pricing['image_cost'] * (image_size_kb / 1000)

    return cost


# Утилиты для списка файлов проекта
def list_project_files(folder_path: str, include: List[str], exclude: List[str], gitignore: pathspec.PathSpec) -> List:
    """Перечисляет файлы проекта согласно фильтрам"""
    _lazy_import_models ()
    files = build_file_tree (folder_path, folder_path, gitignore, include, exclude).folder_content
    files = remove_empty_folders (files)
    files = sort_file_data_alphabetically (files)
    return files


def build_file_tree(root_directory: str, directory: str, gitignore_spec: pathspec.PathSpec,
                    include_patterns: List[str], exclude_patterns: List[str]):
    """Строит дерево файлов проекта с учетом фильтров"""
    _lazy_import_models ()
    entries = os.listdir (directory)
    folder_content = []

    for entry in entries:
        full_path = os.path.join (directory, entry)
        relative_path = os.path.relpath (full_path, start=root_directory)

        # Проверяем правила .gitignore
        if gitignore_spec.match_file (relative_path):
            continue

        if os.path.isdir (full_path):
            folder_node = build_file_tree (root_directory, full_path, gitignore_spec, include_patterns, exclude_patterns)
            folder_content.append (folder_node)
        else:
            match_include = any (glob_match (relative_path, pattern) for pattern in include_patterns)
            match_exclude = any (glob_match (relative_path, pattern) for pattern in exclude_patterns)
            if not match_include or match_exclude:
                continue

            if not is_text_file (full_path):
                continue

            file_node = _FileNode (entry, False, [], 0, 0)
            folder_content.append (file_node)

    return _FileNode (os.path.basename (directory), True, folder_content, 0, 0)


def glob_match(name: str, pat: str) -> bool:
    """Проверяет, соответствует ли имя файла шаблону glob"""
    return glob.globmatch (name, pat, flags=glob.GLOBSTAR)


def remove_empty_folders(file_data_array: List) -> List:
    """Удаляет пустые папки из дерева файлов"""

    def is_not_empty_folder(file_node) -> bool:
        if file_node.is_folder:
            file_node.folder_content = remove_empty_folders (file_node.folder_content)
            return len (file_node.folder_content) > 0
        return True

    return list (filter (is_not_empty_folder, file_data_array))


def sort_file_data_alphabetically(files_data: List) -> List:
    """Сортирует файлы по алфавиту с приоритетом для папок"""
    # Сначала сортируем текущий уровень по алфавиту, папки первыми
    files_data.sort (key=lambda x: (not x.is_folder, x.name))

    # Затем рекурсивно сортируем содержимое каждой папки
    for file_data in files_data:
        if file_data.is_folder:
            file_data.folder_content = sort_file_data_alphabetically (file_data.folder_content)

    return files_data


def compute_sizes(base_path: str, files: List, remove_comments: bool, current_path: str = '') -> int:
    """Вычисляет размеры файлов и количество токенов"""
    total_size = 0
    for file in files:
        full_path = os.path.join (base_path, current_path, file.name)
        if file.is_folder:
            file.size = compute_sizes (base_path, file.folder_content, remove_comments, os.path.join (current_path, file.name))
        else:
            try:
                if not remove_comments:
                    file.size = os.path.getsize (full_path)
                    file.tokens = int (file.size / 4.1)  # Приблизительный подсчет токенов
                else:
                    with open (full_path, 'r', encoding='utf-8') as f:
                        content = f.read ()
                    content = remove_comments (file.name, content)
                    file.size = len (content)
                    file.tokens = get_tokens_cnt (content)
            except Exception as e:
                print (f"Ошибка при обработке файла {full_path}: {e}")
                file.size = 0
                file.tokens = 0
        total_size += file.size
    return total_size


def print_file_tree(files_data: List, current_path: str = '', prefix: str = '') -> None:
    """Печатает дерево файлов в консоль"""
    for file in files_data:
        connector = '└── ' if file == files_data[-1] else '├── '
        new_prefix = prefix + (' ' if file == files_data[-1] else '│ ')
        print (f'{current_path}{prefix}{connector}{file.name} ({bytes_to_str (file.size)})')
        if file.is_folder:
            print_file_tree (file.folder_content, current_path, new_prefix)


def get_file_paths(nodes: List, current_path: str = '') -> List[str]:
    """Извлекает плоский список путей к файлам из дерева файлов"""
    file_paths = []
    for node in nodes:
        full_path = current_path + node.name
        if node.is_folder:
            file_paths.extend (get_file_paths (node.folder_content, full_path + '/'))
        else:
            file_paths.append (full_path)
    return file_paths


def get_proj_stat(file_data: List) -> Any:
    """Собирает статистику по проекту на основе дерева файлов"""
    _lazy_import_models ()
    stats = _ProjStat (
        file_count=0,
        total_size=0,
        total_tokens=0,
        large_files=[]
    )

    def traverse_files(files, current_path: str = ''):
        for file in files:
            full_path = current_path + file.name
            if file.is_folder:
                traverse_files (file.folder_content, full_path + '/')
            else:
                stats.file_count += 1
                stats.total_size += file.size
                stats.total_tokens += file.tokens
                if file.size > LARGE_SOURCE_FILE:
                    stats.large_files.append ({'path': full_path, 'size': file.size})

    traverse_files (file_data)
    stats.large_files.sort (key=lambda x: x['size'], reverse=True)
    return stats


def print_proj_stat(proj_stat) -> None:
    """Печатает статистику проекта в консоль"""
    print ('Сводка по проекту:')
    print (f'Общее количество файлов: {proj_stat.file_count}')
    print (f'Общий размер файлов: {bytes_to_str (proj_stat.total_size)}')
    print (f'Общее количество токенов: {proj_stat.total_tokens}')
    print (f'Большие файлы включены (> {bytes_to_str (LARGE_SOURCE_FILE)}):')
    if not proj_stat.large_files:
        print (' Нет больших файлов')
    else:
        for index, file in enumerate (proj_stat.large_files):
            print (f' {index + 1}. {file["path"]} - Размер: {bytes_to_str (file["size"])}')


# Утилиты для состояния проекта
def load_proj_state(proj_folder: str) -> Any:
    """Загружает состояние проекта из файла"""
    _lazy_import_models ()
    path = os.path.join (DATA_ROOT, proj_folder, 'proj_state.json')

    if not os.path.exists (path):
        # Создать пустое состояние
        return _ProjState (remove_comments=False, files=[])

    with open (path, 'r', encoding='utf-8') as file:
        data = json.load (file)

    return _ProjState.model_validate (data)


def save_proj_state(proj_state, proj_folder: str) -> None:
    """Сохраняет состояние проекта в файл"""
    path = os.path.join (DATA_ROOT, proj_folder, 'proj_state.json')

    # Создаем папку, если она не существует
    os.makedirs (os.path.dirname (path), exist_ok=True)

    # Форматируем JSON для более компактного хранения эмбеддингов
    proj_state_json = json.dumps (proj_state.model_dump (), ensure_ascii=False, indent=4)

    proj_state_json = reformat_proj_state_json (proj_state_json)

    with open (path, 'w', encoding='utf-8') as file:
        file.write (proj_state_json)


def reformat_proj_state_json(json_string: str) -> str:
    """Оптимизирует JSON для более компактного хранения эмбеддингов"""
    # Шаблон для поиска элементов массива
    array_pattern = re.compile (r'"embed"\s*:\s*\[\s*(.*?)\s*\]', re.DOTALL)

    # Функция для замены найденных массивов однострочным форматом
    def replace_array(match):
        array_content = match.group (1)
        # Заменяем переносы строк и лишние пробелы внутри массивов
        single_line_array = array_content.replace ('\n', '')
        single_line_array = re.sub (r'\s+', ' ', single_line_array)
        return f'"embed": [{single_line_array}]'

    # Применяем замену ко всей строке
    return array_pattern.sub (replace_array, json_string)


# Утилиты для работы с семантическим поиском
def find_files_semantic(query: str, app_state, page: int = 0, page_size: int = 10) -> List[str]:
    """
    Выполняет семантический поиск файлов по запросу

    Args:
        query: строка запроса
        app_state: состояние приложения с доступом к модели и файлам
        page: номер страницы результатов
        page_size: размер страницы результатов

    Returns:
        Список путей к файлам, отсортированных по релевантности
    """
    model = SentenceTransformer (app_state.app_config.embedding_model_path)
    ref_embed = model.encode ([query], convert_to_tensor=False)[0]

    ref_embed_array = np.array (ref_embed).reshape (1, -1)
    sim_res = []

    for file in app_state.proj_state.files:
        if file.embed and len (file.embed) > 0:
            file_embed_array = np.array (file.embed).reshape (1, -1)
            similarity_score = cosine_similarity (ref_embed_array, file_embed_array)[0][0]
            sim_res.append ((similarity_score, file))

    sim_res.sort (key=lambda x: x[0], reverse=True)
    page_start = page * page_size
    page_end = page_start + page_size
    page_res = sim_res[page_start:page_end]

    return [add_path_prefix (item[1].path) for item in page_res]


def add_path_prefix(path: str) -> str:
    """Добавляет префикс пути для отображения в чате"""
    return '.' + os.sep + path


def remove_path_prefix(path: str) -> str:
    """Удаляет префикс пути, добавленный для отображения в чате"""
    if path.startswith ('.' + os.sep):
        return path[2:]
    return path


def load_file_content(file_path: str, proj_path: str, remove_comments_flag: bool = False) -> str:
    """Загружает содержимое файла и при необходимости удаляет комментарии"""
    full_path = os.path.join (proj_path, file_path)
    with open (full_path, 'r', encoding='utf-8') as file:
        content = file.read ()

    if remove_comments_flag:
        content = remove_comments (file_path, content)

    return trim_code (content)


def find_in_files(query: str, is_case_sensitive: bool, app_state, page: int = 0,
                  page_size: int = 10, max_lines: int = 5) -> List[Dict[str, Any]]:
    """
    Ищет текстовую строку в файлах проекта

    Args:
        query: строка для поиска
        is_case_sensitive: учитывать ли регистр
        app_state: состояние приложения
        page: номер страницы
        page_size: размер страницы
        max_lines: максимальное количество строк результата на файл

    Returns:
        Список файлов с найденными совпадениями
    """
    results = []

    for file in app_state.proj_state.files:
        try:
            content = load_file_content (file.path, app_state.proj_config.path, app_state.proj_config.remove_comments)
            lines = content.split ('\n')
            matched_lines = []

            for index, line in enumerate (lines):
                if line_matches (line, query, is_case_sensitive):
                    line_number = str (index + 1).ljust (6)
                    matched_lines.append (f"{line_number}{line}")

            if matched_lines:
                if len (matched_lines) > max_lines:
                    delta = len (matched_lines) - max_lines
                    matched_lines = matched_lines[:max_lines] + [f"и еще {delta} совпадений..."]

                results.append ({
                    'path': file.path,
                    'occurrences': matched_lines,
                })
        except Exception as e:
            print (f"Ошибка при поиске в файле {file.path}: {e}")

    # Возвращаем страницу результатов
    start_idx = page * page_size
    end_idx = start_idx + page_size
    return results[start_idx:end_idx]


def line_matches(line: str, query: str, is_case_sensitive: bool) -> bool:
    """Проверяет, содержит ли строка запрос с учетом регистра"""
    if is_case_sensitive:
        return query in line
    else:
        return query.lower () in line.lower ()



