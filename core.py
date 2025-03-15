"""
Основная логика приложения PromptFusion

Содержит функции для инициализации проекта, анализа файлов
и обеспечения интерфейса чата с кодовой базой.
"""

import os
import json
import pathspec
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer

from models import AppState, ProjConfig, ProjState, FileState, ChatSession, ProjectStructure
from config import (
    AppConfig, ProjConfig, load_app_config, save_app_config,
    load_proj_config, save_proj_config, ensure_data_folders,
    is_valid_desc_mode, MODE_DESC, MODE_DESC_NO, MODE_DESC_2,
    DATA_ROOT, LARGE_SOURCE_FILE
)
from utils import (
    input_yes_no, is_yes, is_no, is_folder_exist, load_gitignore,
    list_project_files, compute_sizes, print_file_tree, get_file_paths,
    get_proj_stat, print_proj_stat, load_proj_state, save_proj_state,
    get_tokens_cnt, limit_string, remove_comments, trim_code, get_cost,
    bytes_to_str, add_path_prefix, remove_path_prefix, load_file_content,
    find_files_semantic, find_in_files
)
from analyzers import ProjectAnalyzer, CodeAnalyzer

# Словарь для кэширования эмбеддингов файлов
_embeddings_cache = {}

# Константы
LARGE_SOURCE_FILE_WARNING = 100000  # 100 KB
LARGE_PROJECT_FILES_WARNING = 500  # Файлов
LARGE_PROJECT_SIZE_WARNING = 1000000  # 1 MB
FIND_PAGE_SIZE = 10  # Количество результатов на странице
FIND_MAX_LINES = 5  # Максимальное количество строк на результат


def initialize_project(app_state: AppState) -> bool:
    """
    Инициализирует проект, загружает/создает конфигурации,
    сканирует файлы проекта.

    Returns:
        bool: True если инициализация успешна, False если прервана пользователем
    """
    ensure_data_folders ()

    # Определение проекта
    if app_state.app_config.proj_folder == '':
        # Новый проект
        project_path = enter_project_path ()
        is_new_project = set_current_project (app_state.app_config, project_path)
    else:
        # Проверка существующего проекта
        print ()
        proj_config = load_proj_config (app_state.app_config.proj_folder)
        same_project = input_yes_no (f'Продолжить работу с проектом {proj_config.path}? [Y/n]: ')
        if is_no (same_project):
            project_path = enter_project_path ()
            is_new_project = set_current_project (app_state.app_config, project_path)
        else:
            is_new_project = False

    # Загрузка конфигурации проекта
    proj_config = load_proj_config (app_state.app_config.proj_folder)

    # Проверка режима описания
    if not is_valid_desc_mode (proj_config.desc_mode):
        print (f'Ошибка: Неверное значение desc_mode: {proj_config.desc_mode}')
        return False

    # Загрузка gitignore
    gitignore = pathspec.PathSpec.from_lines ('gitwildmatch', [])
    if proj_config.gitignore:
        gitignore = load_gitignore (proj_config.path)

    # Сканирование файлов проекта
    files = list_project_files (proj_config.path, proj_config.include, proj_config.exclude, gitignore)
    compute_sizes (proj_config.path, files, proj_config.remove_comments)

    print ('\nФайлы, которые будут включены:')
    print_file_tree (files)

    # Вывод статистики проекта
    project_stat = get_proj_stat (files)
    print ()
    print_proj_stat (project_stat)

    # Проверка размера проекта и предупреждения
    check_project_warnings (project_stat)

    # Подтверждение продолжения
    cont_next = input_yes_no ('\nПродолжить с этими настройками проекта? [Y/n]: ')
    if is_no (cont_next):
        if is_new_project:
            print (f'\nРедактируйте файл конфигурации проекта и перезапустите приложение.')
        else:
            print (f'\nРедактируйте конфигурацию и перезапустите приложение.')
        return False

    # Записываем данные в состояние приложения
    app_state.proj_config = proj_config
    app_state.file_paths = get_file_paths (files)
    app_state.proj_stat = project_stat

    return True


def check_project_warnings(project_stat) -> None:
    """Выводит предупреждения о размере проекта, если необходимо"""
    warnings = []
    if project_stat.file_count > LARGE_PROJECT_FILES_WARNING:
        warnings.append (f'Внимание: Общее количество файлов > {LARGE_PROJECT_FILES_WARNING}')
    if project_stat.total_size > LARGE_PROJECT_SIZE_WARNING:
        warnings.append (f'Внимание: Общий размер файлов > {bytes_to_str (LARGE_PROJECT_SIZE_WARNING)}')
    if project_stat.large_files and project_stat.large_files[0]["size"] > LARGE_SOURCE_FILE_WARNING:
        warnings.append (f'Внимание: Есть файл размером > {bytes_to_str (LARGE_SOURCE_FILE_WARNING)}')

    if warnings:
        print ()
        for warning in warnings:
            print (warning)
        print (
            '\nПожалуйста, рассмотрите возможность исключения большего количества файлов. '
            'Это может снизить затраты на API и улучшить качество ответов.'
        )


def enter_project_path() -> str:
    """Запрашивает у пользователя путь к проекту"""
    project_path = ''
    while True:
        project_path = input ('\nВведите путь к проекту: ').strip ()
        if not is_folder_exist (project_path):
            print (f'Ошибка: Путь к проекту не существует: {project_path}')
            continue
        break
    return project_path


def set_current_project(app_config: AppConfig, project_path: str) -> bool:
    """
    Устанавливает текущий проект

    Args:
        app_config: конфигурация приложения
        project_path: путь к проекту

    Returns:
        bool: True если создан новый проект, False если использован существующий
    """
    # Проверяем, существует ли уже проект с таким путем
    existing_proj_folder = find_project_folder (DATA_ROOT, project_path)
    if existing_proj_folder is None:
        # Создаем новый проект
        proj_name = os.path.basename (project_path)
        proj_folder = find_available_proj_folder (DATA_ROOT, proj_name)

        # Создаем новую конфигурацию проекта
        proj_config = ProjConfig ()
        proj_config.path = project_path
        proj_config.include = app_config.default_project_include
        proj_config.exclude = app_config.default_project_exclude
        proj_config.gitignore = app_config.default_project_gitignore
        proj_config.remove_comments = app_config.default_project_remove_comments
        proj_config.desc_mode = app_config.default_project_desc_mode

        # Создаем папку для данных проекта и сохраняем конфигурацию
        ensure_data_folders (proj_folder)
        save_proj_config (proj_config, proj_folder)

        # Обновляем конфигурацию приложения
        app_config.proj_folder = proj_folder
        save_app_config (app_config)

        return True
    else:
        # Используем существующий проект
        app_config.proj_folder = existing_proj_folder
        save_app_config (app_config)
        return False


def find_project_folder(data_folder_path: str, project_path_to_find: str) -> Optional[str]:
    """
    Ищет папку с данными проекта по его пути

    Args:
        data_folder_path: путь к корневой папке данных
        project_path_to_find: путь к проекту для поиска

    Returns:
        Optional[str]: имя папки с данными проекта или None, если не найдено
    """
    if not os.path.exists (data_folder_path):
        return None

    directories = [d for d in os.listdir (data_folder_path) if os.path.isdir (os.path.join (data_folder_path, d))]

    for dir_name in directories:
        config_path = os.path.join (data_folder_path, dir_name, 'proj_config.json')
        if os.path.exists (config_path):
            try:
                with open (config_path, 'r') as f:
                    config_data = json.load (f)
                    if config_data.get ('path') == project_path_to_find:
                        return dir_name
            except:
                continue

    return None


def find_available_proj_folder(base_dir: str, base_name: str) -> str:
    """
    Находит доступное имя для папки проекта

    Args:
        base_dir: базовая директория
        base_name: базовое имя проекта

    Returns:
        str: доступное имя для папки проекта
    """
    counter = 2
    folder_name = base_name

    while os.path.exists (os.path.join (base_dir, folder_name)):
        folder_name = f'{base_name}{counter}'
        counter += 1

    return folder_name


def analyze_project_files(app_state: AppState) -> bool:
    """
    Анализирует файлы проекта, создает описания и эмбеддинги

    Args:
        app_state: состояние приложения

    Returns:
        bool: True если анализ успешен, False если прерван пользователем
    """
    print ("\nАнализ и описание файлов проекта...")
    print ("Вы можете прервать процесс (Ctrl-C) и перезапустить без потери результатов.")

    # Загружаем или создаем состояние проекта
    proj_state_path = os.path.join (DATA_ROOT, app_state.app_config.proj_folder, 'proj_state.json')
    if not os.path.exists (proj_state_path):
        proj_state = ProjState (remove_comments=app_state.proj_config.remove_comments, files=[])
        save_proj_state (proj_state, app_state.app_config.proj_folder)
    else:
        proj_state = load_proj_state (app_state.app_config.proj_folder)

    # Если настройка remove_comments изменилась, сбрасываем кэш
    if proj_state.remove_comments != app_state.proj_config.remove_comments:
        proj_state.remove_comments = app_state.proj_config.remove_comments
        proj_state.files = []
        print ("\nНастройка remove_comments изменена, обновляем все файлы...")

    # Удаляем файлы, которых больше нет в проекте
    prev_len = len (proj_state.files)
    proj_state.files = [file for file in proj_state.files if file.path in app_state.file_paths]
    if prev_len != len (proj_state.files):
        save_proj_state (proj_state, app_state.app_config.proj_folder)

    # Анализируем структуру проекта
    print ("\nАнализ структуры проекта...")
    project_analyzer = ProjectAnalyzer (app_state.proj_config.path, app_state.file_paths)
    project_structure = project_analyzer.analyze_project ()
    proj_state.structure = ProjectStructure (
        class_count=project_structure['class_count'],
        function_count=project_structure['function_count'],
        file_types=project_structure['file_types'],
        external_dependencies=project_structure['external_dependencies'],
        internal_dependencies=project_structure['internal_dependencies'] if 'internal_dependencies' in project_structure else {}
    )

    # Обрабатываем каждый файл
    for file_path in app_state.file_paths:
        full_path = os.path.join (app_state.proj_config.path, file_path)
        mtime = get_file_mtime (full_path)

        # Ищем файл в существующем состоянии или создаем новый
        file_state = next ((f for f in proj_state.files if f.path == file_path), None)
        if not file_state or file_state.mtime != mtime:
            file_state = FileState (path=file_path, mtime=mtime, desc='', desc2='', embed=[])

        content = None

        # Обработка для режима стандартных описаний
        if app_state.proj_config.desc_mode == MODE_DESC and file_state.desc == '':
            content = get_file_content (content, file_path, app_state)
            sys_prompt = get_desc_prompt (get_words_count (len (content)), os.path.basename (file_path))
            file_state.desc = generate_description (app_state, sys_prompt, content)

        # Обработка для режима сокращенных описаний
        if app_state.proj_config.desc_mode == MODE_DESC_2 and file_state.desc2 == '':
            content = get_file_content (content, file_path, app_state)
            sys_prompt = get_desc_prompt_short (get_words_count (len (content)) // 2)
            file_state.desc2 = generate_description (app_state, sys_prompt, content)

        # Генерация эмбеддингов
        if not file_state.embed or len (file_state.embed) == 0:
            content = get_file_content (content, file_path, app_state)
            file_state.embed = generate_embedding (app_state, content)

        # Структурный анализ файла
        analyzer = CodeAnalyzer (os.path.join (app_state.proj_config.path, file_path))
        file_structure = analyzer.parse ()
        if file_structure and "error" not in file_structure:
            file_state.structure = file_structure

        # Обновляем состояние проекта
        if content is not None:
            # Удаляем старую версию файла и добавляем новую
            proj_state.files = [f for f in proj_state.files if f.path != file_path]
            proj_state.files.append (file_state)
            # Сортируем файлы в том же порядке, что и в app_state.file_paths
            proj_state.files = sorted (
                proj_state.files,
                key=lambda f: app_state.file_paths.index (f.path) if f.path in app_state.file_paths else float ('inf')
            )
            # Сохраняем состояние
            save_proj_state (proj_state, app_state.app_config.proj_folder)

    print ("\nАнализ файлов завершен")
    app_state.proj_state = proj_state
    return True


def get_file_mtime(file_path: str) -> int:
    """Возвращает время последней модификации файла"""
    stats = os.stat (file_path)
    return int (stats.st_mtime)


def get_file_content(content: Optional[str], file_path: str, app_state: AppState) -> str:
    """
    Загружает содержимое файла, при необходимости удаляя комментарии

    Args:
        content: существующее содержимое (если есть)
        file_path: путь к файлу
        app_state: состояние приложения

    Returns:
        str: содержимое файла
    """
    if content is not None:
        return content

    print (f"\nЗагрузка файла: {file_path}")
    full_path = os.path.join (app_state.proj_config.path, file_path)

    try:
        with open (full_path, 'r', encoding='utf-8') as file:
            content = file.read ()

        if app_state.proj_config.remove_comments:
            content = remove_comments (file_path, content)

        return trim_code (content)
    except Exception as e:
        print (f"Ошибка при чтении файла {file_path}: {e}")
        return ""


def get_desc_prompt(words: int, name: str) -> str:
    """
    Возвращает промпт для создания описания файла

    Args:
        words: количество слов в описании
        name: имя файла

    Returns:
        str: промпт для LLM
    """
    return (
        f"Создайте очень сжатое ({words} слов) описание файла программного проекта {name}, "
        f"если это код надо кроме описания создать список классов и классовых методов, "
        f"а так же список функций, если это файл контракта то надо подробно описать бизнес логику, "
        f"которая в нем реализована, а так же это сам основной файл контракта или вспомогательный "
        f"(к примеру интерфейс), а так же перечислить публичные методы и что они делают и зачем. "
        f"Отвечай на русском языке."
    )


def get_desc_prompt_short(words: int) -> str:
    """
    Возвращает промпт для создания краткого описания файла

    Args:
        words: количество слов в описании

    Returns:
        str: промпт для LLM
    """
    return (
        f"Создайте очень сжатое ({words} слов), однострочное описание файла, "
        f"если это код надо кроме описания создать список классов и классовых методов, "
        f"а так же список функций. Отвечай на русском языке."
    )


def get_words_count(size: int) -> int:
    """
    Определяет количество слов для описания в зависимости от размера файла

    Args:
        size: размер файла в байтах

    Returns:
        int: рекомендуемое количество слов
    """
    if size < 300:
        return 20
    elif size < 1200:
        return 30
    elif size < 5000:
        return 40
    elif size < 20000:
        return 50
    else:
        return 60


def generate_description(app_state: AppState, sys_prompt: str, content: str) -> str:
    """
    Генерирует описание файла с использованием LLM

    Args:
        app_state: состояние приложения
        sys_prompt: системный промпт для LLM
        content: содержимое файла

    Returns:
        str: сгенерированное описание
    """
    if content == '':
        return 'Файл пуст'

    # Ограничиваем размер входных данных
    content2 = limit_string (content, 15000)
    print ('Создание описания файла, ожидание ответа модели...')

    try:
        # print ('completion 1')
        #
        # print (f'completion 1 app_state.app_config.description_model: {app_state.app_config.description_model}')
        # print (f'completion 1 app_state.app_config.content2: {content2}')

        completion = app_state.openai.chat.completions.create (
            model=app_state.app_config.description_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content2}
            ],
            temperature=0.2
        )
        # print ('completion 2')
        desc = completion.choices[0].message.content
        print (f'Описание создано: {desc[:50]}...')
        return desc
    except Exception as e:
        print (f"Ошибка при создании описания: {e}")
        return "Не удалось создать описание файла из-за ошибки API"


def generate_embedding(app_state: AppState, content: str) -> List[float]:
    """
    Генерирует эмбеддинг для содержимого файла

    Args:
        app_state: состояние приложения
        content: содержимое файла

    Returns:
        List[float]: эмбеддинг содержимого
    """
    if content == '':
        return []

    # Ограничиваем размер входных данных
    content2 = limit_string (content, 8000)
    print ('Создание эмбеддинга...')

    # Используем кэш, если возможно
    content_hash = hash (content2)
    if content_hash in _embeddings_cache:
        print ('Эмбеддинг найден в кэше')
        return _embeddings_cache[content_hash]

    try:
        model = SentenceTransformer (app_state.app_config.embedding_model_path)
        embeddings = model.encode ([content2], convert_to_tensor=False)
        embedding = embeddings[0].tolist ()

        # Сохраняем в кэш
        _embeddings_cache[content_hash] = embedding

        print ('Эмбеддинг создан')
        return embedding
    except Exception as e:
        print (f"Ошибка при создании эмбеддинга: {e}")
        return []


def start_chat_session(app_state: AppState) -> None:
    """
    Запускает интерактивный чат с LLM для обсуждения проекта

    Args:
        app_state: состояние приложения
    """
    session = ChatSession ()

    # Добавляем системное сообщение
    sys_prompt = get_sys_prompt (app_state)
    sys_prompt_tokens = get_tokens_cnt (sys_prompt)
    sys_prompt_cost = get_cost (app_state.app_config.chat_model, sys_prompt_tokens, 0)

    print (f"\nСистемный промпт: стоимость ввода: ${sys_prompt_cost:.4f} ({sys_prompt_tokens} токенов)")
    print (f"\nТеперь вы можете задавать вопросы о проекте {os.path.basename (app_state.proj_config.path)}.")
    print ("Нажмите Ctrl-C для прерывания в любой момент.")

    while not session.exit_requested:
        try:
            # Получаем ввод пользователя
            user_input = get_user_input (session)
            if session.exit_requested:
                break

            # Формируем сообщения для запроса
            messages = session.messages + [
                {'role': 'user', 'content': user_input},
                {'role': 'system', 'content': sys_prompt}
            ]

            # Отправляем запрос к LLM
            print ("\nОжидание ответа модели...")

            try:
                response = app_state.openai.chat.completions.create (
                    model=app_state.app_config.chat_model,
                    messages=messages,
                    temperature=0.1
                )
            except Exception as e:
                print (f"Ошибка при запросе ... к API: {e}")
                continue

            # Обрабатываем ответ
            if not hasattr (response, 'usage') or response.usage is None:
                print (f"Usage: app_state.app_config.chat_model {app_state.app_config.chat_model}")
                print ("Ошибка: response.usage вернул None. Возможные причины: неудачный API-запрос или проблемы с авторизацией.")
                continue


            # Обновляем статистику токенов
            session.request_in_tokens = response.usage.prompt_tokens
            session.request_out_tokens = response.usage.completion_tokens
            session.total_in_tokens += response.usage.prompt_tokens
            session.total_out_tokens += response.usage.completion_tokens

            # Получаем сообщение ответа
            response_message = response.choices[0].message

            # Добавляем сообщение пользователя в историю
            session.messages.append ({'role': 'user', 'content': user_input})

            # Проверяем наличие вызовов инструментов
            if response_message.tool_calls is None:
                # Если нет вызовов инструментов, просто выводим ответ
                session.messages.append (response_message.model_dump (exclude_none=True))

                # Выводим статистику
                chat_tokens = sum (get_message_tokens (message) for message in session.messages)
                chat_cost = get_cost (app_state.app_config.chat_model, chat_tokens, 0)

                request_cost_in = get_cost (app_state.app_config.chat_model, session.request_in_tokens, 0)
                request_cost_out = get_cost (app_state.app_config.chat_model, 0, session.request_out_tokens)
                total_cost = get_cost (
                    app_state.app_config.chat_model,
                    session.total_in_tokens,
                    session.total_out_tokens
                )

                print ()
                print (f"Стоимость следующего ввода: ${(sys_prompt_cost + chat_cost):.4f} "
                       f"(системный: ${sys_prompt_cost:.4f}, чат: ${chat_cost:.4f})")
                print (f"Стоимость запроса: ${(request_cost_in + request_cost_out):.4f} "
                       f"(ввод: ${request_cost_in:.4f}, вывод: ${request_cost_out:.4f})")
                print (f"Общая стоимость сессии: ${total_cost:.4f}")

                print ()
                print (f'🤖 Бот: {response_message.content}')

                # Сбрасываем счетчики для следующего запроса
                session.request_in_tokens = 0
                session.request_out_tokens = 0
            else:
                # Если есть контент, выводим его
                if response_message.content:
                    print ()
                    print (f'🤖 Бот: {response_message.content}')

                # Обрабатываем вызовы инструментов
                for tool_call in response_message.tool_calls:
                    result = call_function (tool_call.function, app_state)

                    # Добавляем результат в историю сообщений
                    session.messages.append (response_message.model_dump (exclude_none=True))
                    session.messages.append ({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': json.dumps (result),
                    })

                # После обработки инструментов, отправляем новый запрос
                continue

        except KeyboardInterrupt:
            print ("\nЧат прерван пользователем")
            break


def get_user_input(session: ChatSession) -> str:
    """
    Получает ввод пользователя и обрабатывает команды

    Args:
        session: сессия чата

    Returns:
        str: ввод пользователя или пустая строка, если выполнена команда
    """
    while True:
        print ()
        user_input = input ('👤 Вы: ').strip ()

        # Проверяем команды
        if user_input.lower () == '/exit':
            session.exit_requested = True
            return ""
        elif user_input.lower () == '/clear':
            session.messages = []
            print ("\nИстория сообщений очищена")
            continue

        return user_input


def get_message_tokens(message: Dict[str, str]) -> int:
    """
    Подсчитывает количество токенов в сообщении

    Args:
        message: сообщение в формате {'role': '...', 'content': '...'}

    Returns:
        int: количество токенов
    """
    content = message.get ('content')
    if content is not None:
        return get_tokens_cnt (content)
    return 0


#def get_sys_prompt(app_state: AppState) -> str:

def get_sys_prompt(app_state: AppState) -> str:
    """
    Формирует системный промпт для чата в зависимости от режима описания

    Args:
        app_state: состояние приложения

    Returns:
        str: системный промпт
    """
    desc_mode = app_state.proj_config.desc_mode

    if desc_mode == MODE_DESC:
        proj_struct = get_sys_context_desc (app_state.proj_state.files)
        return get_sys_prompt_template (proj_struct, app_state.proj_state.structure, True)
    elif desc_mode == MODE_DESC_NO:
        proj_struct = get_sys_context_no_desc (app_state.proj_state.files)
        return get_sys_prompt_template (proj_struct, app_state.proj_state.structure, False)
    elif desc_mode == MODE_DESC_2:
        proj_struct = get_sys_context_short_desc (app_state.proj_state.files)
        return get_sys_prompt_template (proj_struct, app_state.proj_state.structure, True)
    else:
        raise ValueError (f"Неверный режим описания: {desc_mode}")


def get_sys_prompt_template(proj_struct: str, project_structure: ProjectStructure, is_desc: bool) -> str:
    """
    Шаблон системного промпта

    Args:
        proj_struct: строка с описанием файлов проекта
        project_structure: структура проекта
        is_desc: есть ли описания файлов

    Returns:
        str: системный промпт
    """
    with_descs = " с описаниями" if is_desc else ""

    structure_info = ""
    if project_structure:
        structure_info = f"""
Общая структура проекта:
- Количество классов: {project_structure.class_count}
- Количество функций: {project_structure.function_count}
- Типы файлов: {', '.join ([f"{k}: {v}" for k, v in project_structure.file_types.items ()])}
- Внешние зависимости: {', '.join (project_structure.external_dependencies)}
"""

    return f"""Вы чат-бот, задача которого отвечать на вопросы пользователя на русском языке о программном проекте.

{structure_info}

Ниже приведен список всех файлов проекта{with_descs}.

Если нужно загрузить содержимое файла, используйте функцию "get_file".
Вы можете искать файлы с помощью функции 'find_files_semantic'.
Также можно искать в содержимом файлов с помощью функции 'find_in_files'.
Вас также могут попросить внести изменения в проект: используйте функцию update_file для создания или обновления файла.

Файлы проекта{with_descs}:

{proj_struct}
"""


def get_sys_context_desc(files: List[FileState]) -> str:
    """
    Формирует контекст с полными описаниями файлов

    Args:
        files: список файлов

    Returns:
        str: форматированный список файлов с описаниями
    """
    return '\n\n'.join (f"{add_path_prefix (file.path)}\n{file.desc}" for file in files)


def get_sys_context_no_desc(files: List[FileState]) -> str:
    """
    Формирует контекст без описаний файлов

    Args:
        files: список файлов

    Returns:
        str: форматированный список файлов
    """
    return '\n'.join (f"{add_path_prefix (file.path)}" for file in files)


def get_sys_context_short_desc(files: List[FileState]) -> str:
    """
    Формирует контекст с краткими описаниями файлов

    Args:
        files: список файлов

    Returns:
        str: форматированный список файлов с краткими описаниями
    """
    return '\n\n'.join (f"{add_path_prefix (file.path)}\n{file.desc2}" for file in files)

