"""
Модуль для реализации функций инструментов через промпт-инжиниринг
вместо использования механизма функций API OpenAI/OpenRouter.


"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from models import AppState, FileState
from utils import (
    input_yes_no, is_no, load_file_content, add_path_prefix,
    remove_path_prefix, trim_code, remove_comments
)
import re


def fix_encoding(broken_text):
    """
    Исправляет битую кодировку кириллицы, работая с байтами напрямую.

    Args:
        broken_text: Строка с битой кодировкой

    Returns:
        Исправленная строка
    """
    # Получаем байты через UTF-8 с заменой
    raw_bytes = broken_text.encode('utf-8', errors='replace')
    print(f"Исходные байты: {raw_bytes.hex()}")

    # Проверяем стандартный путь: Latin1 -> UTF-8
    try:
        temp_bytes = broken_text.encode('latin1', errors='replace')
        result = temp_bytes.decode('utf-8', errors='ignore')
        if any(1040 <= ord(c) <= 1103 for c in result):
            print("Успешное преобразование: latin1-encoded UTF-8 -> UTF-8")
            return result
    except UnicodeError as e:
        print(f"Ошибка при декодировании Latin1 -> UTF-8: {str(e)}")

    # Упрощённый подход: собираем только валидные UTF-8 последовательности
    corrected_bytes = bytearray()
    i = 0
    while i < len(raw_bytes):
        # Пропускаем символ замены (efbfbd)
        if i + 2 < len(raw_bytes) and raw_bytes[i:i+3] == b'\xef\xbf\xbd':
            i += 3
            continue
        # Обрабатываем c3xx или c2xx как Latin1-интерпретированный UTF-8
        if i + 1 < len(raw_bytes) and raw_bytes[i] in (0xc3, 0xc2):
            byte2 = raw_bytes[i + 1]
            if 0xa0 <= byte2 <= 0xbf:  # Кириллица в UTF-8 начинается с d0 или d1
                corrected_bytes.extend(bytes([0xd0 + ((byte2 - 0xa0) // 4), 0x80 + (byte2 & 0x3f)]))
            elif 0x80 <= byte2 <= 0x9f:  # Дополнительные символы
                corrected_bytes.extend(bytes([0xd0, byte2]))
            else:
                corrected_bytes.append(raw_bytes[i])
            i += 2
        else:
            corrected_bytes.append(raw_bytes[i])
            i += 1

    print(f"Скорректированные байты: {corrected_bytes.hex()}")

    # Декодируем результат
    try:
        result = corrected_bytes.decode('utf-8', errors='ignore')
        if any(1040 <= ord(c) <= 1103 for c in result):
            print("Успешное восстановление через анализ байтов")
            return result
        else:
            print("Результат не содержит кириллицу")
    except UnicodeError as e:
        print(f"Ошибка при декодировании скорректированных байтов: {str(e)}")

    print("Не удалось исправить кодировку")
    return broken_text


class PromptTools:
    """
    Класс для работы с инструментами через промпт вместо API функций.
    """

    def __init__(self, app_state: AppState):
        """
        Инициализирует инструменты для работы с промптами.

        Args:
            app_state: Состояние приложения с доступом к проекту и моделям
        """
        self.app_state = app_state
        self.find_page_size = 10
        self.find_max_lines = 5

    def get_tools_system_prompt(self) -> str:
        """
        Возвращает дополнительные инструкции для системного промпта
        для работы с функциями через промпт.

        Returns:
            str: Инструкции для работы с функциями
        """
        return """
Вы можете выполнять следующие функции для работы с проектом:

1. get_file(path): Загружает содержимое файла.
   Пример использования: 
   ```
   Мне нужно посмотреть файл main.py
   [FUNCTION: get_file(path="./main.py")]
   ```

2. find_files_semantic(query, page): Поиск файлов с использованием семантического поиска.
   Пример использования:
   ```
   Найди файлы, связанные с обработкой изображений
   [FUNCTION: find_files_semantic(query="обработка изображений", page=0)]
   ```

3. find_in_files(query, is_case_sensitive, page): Поиск текста в файлах.
   Пример использования:
   ```
   Найди все места, где используется функция process_data
   [FUNCTION: find_in_files(query="process_data", is_case_sensitive=false, page=0)]
   ```

4. update_file(path, content): Обновляет или создает файл.
   Пример использования:
   ```
   Хочу создать новый файл utils/helpers.py
   [FUNCTION: update_file(path="./utils/helpers.py", content="# New helpers file\n\ndef helper_function():\n    pass")]
   ```

Когда пользователь просит выполнить подобные действия, используйте точный формат [FUNCTION: имя_функции(параметры)] для указания функции. Дождитесь результата выполнения функции, прежде чем продолжить ответ.
"""

    def enhance_system_prompt(self, original_prompt: str) -> str:
        """
        Добавляет инструкции по работе с инструментами к системному промпту.

        Args:
            original_prompt: Исходный системный промпт

        Returns:
            str: Расширенный системный промпт с инструкциями
        """
        tools_prompt = self.get_tools_system_prompt ()
        return original_prompt + "\n\n" + tools_prompt

    def process_user_message(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Проверяет сообщение пользователя на наличие вызовов функций и выполняет их
        с корректной обработкой кодировки.

        Args:
            message: Сообщение пользователя

        Returns:
            Optional[Dict[str, Any]]: Результат выполнения функции или None
        """
        import re

        # Шаблоны для распознавания вызовов функций в тексте
        function_pattern = r'\[FUNCTION: (\w+)\((.*?)\)\]'

        matches = re.findall (function_pattern, message)
        if not matches:
            return None

        # Берем только первое совпадение
        function_name, args_str = matches[0]

        # Более надежный парсинг аргументов с учетом вложенных структур и экранированных кавычек
        args = {}
        # Используем улучшенный парсер для аргументов
        # Обрабатываем одиночные и двойные кавычки, а также числовые и булевы значения
        args_pattern = r'(\w+)=(?:\"((?:\\.|[^\"])*)\"|\'((?:\\.|[^\'])*?)\'|(\d+\.\d+)|(\d+)|(true|false))'

        for arg_match in re.findall (args_pattern, args_str):
            arg_name = arg_match[0]

            # Находим первое непустое значение среди возможных типов
            arg_values = arg_match[1:]
            arg_value = next ((v for v in arg_values if v), None)

            # Преобразуем значение к правильному типу
            if arg_value == "true":
                arg_value = True
            elif arg_value == "false":
                arg_value = False
            elif arg_value and arg_value.isdigit ():
                arg_value = int (arg_value)
            elif arg_value and arg_value.replace ('.', '', 1).isdigit ():
                arg_value = float (arg_value)

            # Если это строка с экранированием, обрабатываем её с учетом unicode
            if isinstance (arg_value, str) and ('\\' in arg_value or '\\u' in arg_value or '\\x' in arg_value):
                try:
                    # Прямой подход через codecs для разбора unicode escape-последовательностей
                    import codecs
                    arg_value = codecs.decode (arg_value, 'unicode_escape')
                except Exception as e:
                    print (f"Ошибка декодирования Unicode в аргументе: {e}")
                    try:
                        # Альтернативный метод - используем буфер для замены escape-последовательностей
                        # Это безопаснее чем encode/decode
                        import re

                        # Заменяем \uXXXX последовательности на их символы
                        def replace_unicode(match):
                            hex_val = match.group (1)
                            return chr (int (hex_val, 16))

                        # Заменяем \n, \t и другие common escapes
                        arg_value = arg_value.replace ('\\n', '\n').replace ('\\t', '\t').replace ('\\"', '"')
                        # Заменяем \uXXXX последовательности
                        arg_value = re.sub (r'\\u([0-9a-fA-F]{4})', replace_unicode, arg_value)
                    except Exception as e2:
                        print (f"Вторичная ошибка декодирования: {e2}")
                        # Оставляем как есть если не получилось декодировать

            args[arg_name] = arg_value

        # Вызываем соответствующую функцию
        if function_name == "get_file":
            return self.get_file_func (args.get ("path", ""))
        elif function_name == "find_files_semantic":
            return self.find_files_semantic_func (args.get ("query", ""), args.get ("page", 0))
        elif function_name == "find_in_files":
            return self.find_in_files_func (
                args.get ("query", ""),
                args.get ("is_case_sensitive", False),
                args.get ("page", 0)
            )
        elif function_name == "update_file":
            return self.update_file_func (args.get ("path", ""), args.get ("content", ""))
        else:
            return {"error": f"Неизвестная функция: {function_name}"}


    def get_file_func(self, path: str) -> Dict[str, Any]:
        """
        Загружает содержимое файла.

        Args:
            path: Путь к файлу

        Returns:
            Dict[str, Any]: Результат выполнения с содержимым файла или ошибкой
        """
        print (f'\nВызов функции get_file, path: {path}')

        clean_path = remove_path_prefix (path)
        if clean_path not in self.app_state.file_paths:
            print (f'\n!!!!!!!!!!!!!!!!!!! неверный путь: {path}')
            return {
                'path': path,
                'error': 'Неверный путь к файлу',
                'function': 'get_file'
            }

        try:
            content = load_file_content (
                clean_path,
                self.app_state.proj_config.path,
                self.app_state.proj_config.remove_comments
            )
            print (f'возвращено содержимое файла {path}')

            return {
                'path': path,
                'content': content,
                'function': 'get_file'
            }
        except Exception as e:
            print (f"\nОшибка при загрузке файла {path}: {e}")
            return {
                'path': path,
                'error': f'Ошибка при загрузке файла: {str (e)}',
                'function': 'get_file'
            }

    def find_files_semantic_func(self, query: str, page: int) -> Dict[str, Any]:
        """
        Выполняет семантический поиск файлов.

        Args:
            query: Строка запроса
            page: Номер страницы результатов

        Returns:
            Dict[str, Any]: Результаты поиска
        """
        print (f"\nВызов функции find_files_semantic, query: {query}, page: {page}")

        try:
            model = SentenceTransformer (self.app_state.app_config.embedding_model_path)
            ref_embed = model.encode ([query], convert_to_tensor=False)[0]

            ref_embed_array = np.array (ref_embed).reshape (1, -1)
            sim_res = []

            for file in self.app_state.proj_state.files:
                if file.embed and len (file.embed) > 0:
                    file_embed_array = np.array (file.embed).reshape (1, -1)
                    similarity_score = cosine_similarity (ref_embed_array, file_embed_array)[0][0]
                    sim_res.append ((similarity_score, file))

            sim_res.sort (key=lambda x: x[0], reverse=True)
            page_start = page * self.find_page_size
            page_end = page_start + self.find_page_size
            page_res = sim_res[page_start:page_end]

            result_paths = [add_path_prefix (item[1].path) for item in page_res]

            print ('результаты:')
            for path in result_paths:
                print (path)

            return {
                'results': result_paths,
                'function': 'find_files_semantic',
                'query': query,
                'page': page
            }
        except Exception as e:
            print (f"\nОшибка при семантическом поиске: {e}")
            return {
                'error': f"Ошибка при поиске: {str (e)}",
                'function': 'find_files_semantic'
            }

    def find_in_files_func(self, query: str, is_case_sensitive: bool, page: int) -> Dict[str, Any]:
        """
        Поиск текста в файлах проекта.

        Args:
            query: Строка поиска
            is_case_sensitive: Учитывать ли регистр
            page: Номер страницы результатов

        Returns:
            Dict[str, Any]: Результаты поиска
        """
        print (f"\nВызов функции find_in_files, query: {query}, isCaseSensitive: {is_case_sensitive}, page: {page}")

        try:
            results = []

            for file in self.app_state.proj_state.files:
                try:
                    content = load_file_content (
                        file.path,
                        self.app_state.proj_config.path,
                        self.app_state.proj_config.remove_comments
                    )
                    lines = content.split ('\n')
                    matched_lines = []

                    for index, line in enumerate (lines):
                        if self._line_matches (line, query, is_case_sensitive):
                            line_number = str (index + 1).ljust (6)
                            matched_lines.append (f"{line_number}{line}")

                    if matched_lines:
                        if len (matched_lines) > self.find_max_lines:
                            delta = len (matched_lines) - self.find_max_lines
                            matched_lines = matched_lines[:self.find_max_lines] + [f"и еще {delta} совпадений..."]

                        results.append ({
                            'path': add_path_prefix (file.path),
                            'occurrences': matched_lines,
                        })
                except Exception as e:
                    print (f"Ошибка при поиске в файле {file.path}: {e}")

            # Возвращаем страницу результатов
            start_idx = page * self.find_page_size
            end_idx = start_idx + self.find_page_size
            page_results = results[start_idx:end_idx]

            print ('результаты:')
            for file in page_results:
                print (file['path'])
                for occurrence in file['occurrences']:
                    print (f"  {occurrence}")

            if not page_results:
                print ('Ничего не найдено')

            return {
                'results': page_results,
                'function': 'find_in_files',
                'query': query,
                'is_case_sensitive': is_case_sensitive,
                'page': page
            }
        except Exception as e:
            print (f"\nОшибка при поиске в файлах: {e}")
            return {
                'error': f"Ошибка при поиске: {str (e)}",
                'function': 'find_in_files'
            }

    def _line_matches(self, line: str, query: str, is_case_sensitive: bool) -> bool:
        """
        Проверяет, содержит ли строка запрос.

        Args:
            line: Строка для проверки
            query: Строка запроса
            is_case_sensitive: Учитывать ли регистр

        Returns:
            bool: True, если строка содержит запрос
        """
        if is_case_sensitive:
            return query in line
        else:
            return query.lower () in line.lower ()

    def update_file_func(self, path: str, content: str) -> Dict[str, Any]:
        """
        Обновляет или создает файл с корректной обработкой кодировки.

        Args:
            path: Путь к файлу
            content: Новое содержимое файла

        Returns:
            Dict[str, Any]: Результат операции
        """
        print (f'\nВызов функции update_file, path: {path}')

        # Запрашиваем подтверждение
        if is_no (input_yes_no (f'Модель хочет обновить файл: {path}, подтверждаете? [Y/n]: ')):
            return {
                'error': f'Пользователь отклонил обновление файла, path: {path}',
                'function': 'update_file'
            }

        clean_path = remove_path_prefix (path)
        full_path = os.path.join (self.app_state.proj_config.path, clean_path)

        # Проверяем существование папки
        folder_path = os.path.dirname (full_path)
        if not os.path.exists (folder_path):
            print (f'\nОШИБКА: Папка не существует, folder_path: {add_path_prefix (folder_path)}')
            return {
                'error': f'Папка не существует, folder_path: {add_path_prefix (folder_path)}',
                'function': 'update_file'
            }

        content = fix_encoding (content)
        # Обработка и декодирование многострочных строк с экранированием

        processed_content = content
        print(f"Processed content: \n {processed_content} \n")

        # Проверка, является ли content строкой с буквальным экранированием
        if '\\n' in content or '\\u' in content or '\\x' in content:
            try:
                # Используем правильное декодирование строки для обработки экранированных последовательностей
                # Включая unicode-escape для обработки \uXXXX последовательностей
                processed_content = content.encode ('latin1').decode ('unicode_escape')
            except Exception as e:
                print (f"Предупреждение при декодировании unicode: {e}")
                try:
                    # Альтернативный метод с использованием eval
                    processed_content = eval (f'"""{content}"""')
                except Exception as e2:
                    print (f"Предупреждение при использовании eval: {e2}")
                    # Простой метод замены
                    processed_content = content.replace ('\\n', '\n').replace ('\\t', '\t').replace ('\\"', '"')

        # Записываем файл явно с UTF-8 кодировкой
        try:
            # Проверка кодировки перед записью
            print (f"Проверка кодировки перед записью: {type (processed_content)}")
            print (f"Processed content 2 : \n {processed_content} \n")

            # Явно конвертируем в UTF-8 при записи
            with open (full_path, 'w', encoding='utf-8') as file:
                file.write (processed_content)

            print (f'файл {path} обновлен')

            # Обновляем состояние файла
            if clean_path in self.app_state.file_paths:
                # Если файл уже был в списке, обновляем его состояние
                mtime = os.stat (full_path).st_mtime
                file_state = next ((f for f in self.app_state.proj_state.files if f.path == clean_path), None)
                if file_state:
                    file_state.mtime = int (mtime)
                    file_state.desc = ''  # Сбрасываем описание
                    file_state.desc2 = ''  # Сбрасываем краткое описание
                    file_state.embed = []  # Сбрасываем эмбеддинг
                    file_state.structure = None  # Сбрасываем структуру
            else:
                # Если это новый файл, добавляем его в список
                self.app_state.file_paths.append (clean_path)

            return {
                'path': path,
                'result': 'Файл обновлен',
                'function': 'update_file'
            }
        except Exception as e:
            print (f"\nОшибка при обновлении файла {path}: {e}")
            return {
                'error': f'Ошибка при обновлении файла: {str (e)}',
                'function': 'update_file'
            }



    def format_function_result(self, result: Dict[str, Any]) -> str:
        """
        Форматирует результат выполнения функции для вставки в промпт.

        Args:
            result: Результат выполнения функции

        Returns:
            str: Форматированный результат
        """
        function_name = result.get ('function', 'unknown')

        if 'error' in result:
            return f"[RESULT: {function_name}]\nОшибка: {result['error']}\n[/RESULT]"

        if function_name == 'get_file':
            return (
                f"[RESULT: {function_name}]\n"
                f"Файл: {result['path']}\n"
                f"Содержимое:\n```\n{result['content']}\n```\n"
                f"[/RESULT]"
            )

        elif function_name == 'find_files_semantic':
            file_list = '\n'.join ([f"- {path}" for path in result['results']])
            return (
                f"[RESULT: {function_name}]\n"
                f"Запрос: {result['query']}\n"
                f"Страница: {result['page']}\n"
                f"Найденные файлы:\n{file_list}\n"
                f"[/RESULT]"
            )

        elif function_name == 'find_in_files':
            result_str = f"[RESULT: {function_name}]\n"
            result_str += f"Запрос: {result['query']}\n"
            result_str += f"Учет регистра: {'Да' if result['is_case_sensitive'] else 'Нет'}\n"
            result_str += f"Страница: {result['page']}\n\n"

            if not result['results']:
                result_str += "Ничего не найдено.\n"
            else:
                for file_result in result['results']:
                    result_str += f"Файл: {file_result['path']}\n"
                    for occurrence in file_result['occurrences']:
                        result_str += f"  {occurrence}\n"
                    result_str += "\n"

            result_str += "[/RESULT]"
            return result_str

        elif function_name == 'update_file':
            return (
                f"[RESULT: {function_name}]\n"
                f"Файл: {result['path']}\n"
                f"Статус: {result['result']}\n"
                f"[/RESULT]"
            )

        # Для неизвестных функций
        return f"[RESULT: {function_name}]\n{json.dumps (result, indent=2)}\n[/RESULT]"


    def process_bot_response(self, response: str) -> str:
        """
        Обрабатывает ответ бота, заменяя вызовы функций на их результаты.

        Args:
            response: Ответ бота

        Returns:
            str: Обработанный ответ с результатами функций
        """
        # Шаблон для поиска вызовов функций с более точным захватом аргументов
        function_pattern = r'\[FUNCTION: (\w+)\((.*?)\)\]'

        # Пока в ответе есть вызовы функций, обрабатываем их
        while re.search(function_pattern, response):
            # Находим первый вызов функции
            match = re.search(function_pattern, response)
            if not match:
                break

            function_call = match.group(0)
            function_name = match.group(1)
            args_str = match.group(2)

            # Создаем сообщение пользователя с вызовом функции
            function_message = f"{function_call}"

            # Выполняем функцию
            result = self.process_user_message(function_message)

            if result:
                # Форматируем результат
                formatted_result = self.format_function_result(result)

                # Заменяем вызов функции на результат
                response = response.replace(function_call, formatted_result, 1)
            else:
                # Если функция не распознана, заменяем вызов на ошибку
                error_result = f"[RESULT: error]\nНе удалось выполнить функцию: {function_name}\n[/RESULT]"
                response = response.replace(function_call, error_result, 1)

        return response


