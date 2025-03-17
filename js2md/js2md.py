#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import logging
import re

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conversion_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)

def normalize_text(text):
    """
    Нормализует текст: удаляет лишние пробелы и переводы строк.
    
    Args:
        text (str): Исходный текст
        
    Returns:
        str: Нормализованный текст
    """
    if not text:
        return text
        
    # Заменяем множественные пробелы на один
    normalized = re.sub(r'\s+', ' ', text)
    # Удаляем пробелы в начале и конце строки
    normalized = normalized.strip()
    
    return normalized

def convert_json_to_md(input_file_path, output_file_path, excluded_extensions=None):
    """
    Преобразует JSON-файл в Markdown-документ.
    
    Args:
        input_file_path (str): Путь к входному JSON-файлу
        output_file_path (str): Путь к выходному MD-файлу
        excluded_extensions (list, optional): Список расширений файлов, которые нужно исключить
    """
    if excluded_extensions is None:
        excluded_extensions = [".css", ".svg", ".scss"]
    
    logging.info(f"Исключаемые расширения: {excluded_extensions}")
    logging.info(f"Начало преобразования: {input_file_path} -> {output_file_path}")
    
    # Проверяем существование входного файла
    if not os.path.exists(input_file_path):
        error_msg = f"Ошибка: Файл {input_file_path} не найден"
        logging.error(error_msg)
        print(error_msg)
        # Возвращаем False, чтобы показать, что конвертация не удалась
        return False
    
    try:
        # Открываем JSON-файл и загружаем данные
        with open(input_file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            logging.debug(f"Содержимое файла: {file_content[:500]}..." if len(file_content) > 500 else file_content)
            
            # Проверяем, не пустой ли файл
            if not file_content.strip():
                logging.error("JSON-файл пуст")
                return
                
            # Предполагаем, что файл может содержать список элементов или один большой объект
            json_data = json.loads(file_content)
            logging.info(f"Тип загруженных данных: {type(json_data).__name__}")
            logging.debug(f"Структура данных: {json_data}")
            
            # Если данные в виде объекта, а не списка, преобразуем в список
            if isinstance(json_data, dict):
                logging.info("Данные в виде словаря (объекта)")
                
                # Проверяем наличие поля "files" со списком объектов
                if "files" in json_data and isinstance(json_data["files"], list):
                    logging.info(f"Найден массив 'files' с {len(json_data['files'])} элементами")
                    json_data = json_data["files"]
                    logging.info("Использован массив из поля 'files'")
                else:
                    # Проверяем наличие вложенных объектов
                    has_nested_dicts = any(isinstance(value, dict) for value in json_data.values())
                    logging.info(f"Содержит вложенные словари: {has_nested_dicts}")
                    
                    if not has_nested_dicts:
                        json_data = [json_data]
                        logging.info("Преобразовано в список из одного объекта")
                    else:
                        # Извлекаем вложенные объекты
                        nested_objects = [value for value in json_data.values() if isinstance(value, dict)]
                        json_data = nested_objects
                        logging.info(f"Извлечено {len(nested_objects)} вложенных объектов")
            
            logging.info(f"Итоговое количество элементов для обработки: {len(json_data)}")
            
            # Фильтруем данные, исключая файлы с определенными расширениями
            filtered_data = []
            for i, item in enumerate(json_data):
                logging.debug(f"Элемент #{i+1}: ключи = {list(item.keys())}")
                
                # Проверяем наличие ключевых полей
                if 'path' not in item:
                    logging.warning(f"Элемент #{i+1} не содержит ключ 'path'")
                    continue
                    
                if 'desc' not in item:
                    logging.warning(f"Элемент #{i+1} не содержит ключ 'desc'")
                
                # Проверяем, нужно ли исключить файл по расширению
                path = item.get('path', '')
                should_exclude = any(path.endswith(ext) for ext in excluded_extensions)
                
                if should_exclude:
                    logging.info(f"Исключен файл с путем: {path}")
                    continue
                
                # Добавляем элемент в отфильтрованный список
                filtered_data.append(item)
            
            # Заменяем исходный список отфильтрованным
            json_data = filtered_data
            logging.info(f"После фильтрации осталось {len(json_data)} элементов")
    
        # Создаем директорию для выходного файла, если она не существует
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Создана директория: {output_dir}")
        
        # Открываем файл для записи Markdown
        with open(output_file_path, 'w', encoding='utf-8') as md_file:
            logging.info("Начало записи Markdown-файла")
            items_written = 0
            
            for i, item in enumerate(json_data):
                logging.debug(f"Обработка элемента #{i+1}")
                has_written_content = False
                
                # Добавляем разделитель между записями, кроме первой
                if i > 0:
                    md_file.write("\n\n---\n\n")
                    logging.debug("Добавлен разделитель между записями")
                else:
                    md_file.write("---\n\n")
                    logging.debug("Добавлен начальный разделитель")
                
                # Добавляем путь к файлу
                if 'path' in item:
                    path = item['path']
                    md_file.write(f"## {path}\n\n")
                    logging.debug(f"Добавлен заголовок: {path}")
                    has_written_content = True
                
                # Добавляем описание с нормализацией текста
                if 'desc' in item and item['desc']:
                    desc = normalize_text(item['desc'])
                    if desc:
                        md_file.write(f"{desc}\n")
                        logging.debug("Добавлено основное описание")
                        has_written_content = True
                
                # Добавляем дополнительное описание с нормализацией, если оно есть
                if 'desc2' in item and item['desc2']:
                    desc2 = normalize_text(item['desc2'])
                    if desc2:
                        md_file.write(f"\n{desc2}\n")
                        logging.debug("Добавлено дополнительное описание")
                        has_written_content = True
                
                if has_written_content:
                    items_written += 1
            
            logging.info(f"Записано {items_written} элементов из {len(json_data)}")
        
        if items_written > 0:
            success_msg = f"Преобразование завершено успешно. Markdown-файл сохранен: {output_file_path}"
            logging.info(success_msg)
            print(success_msg)
            return True
        else:
            warning_msg = f"Преобразование завершено, но в выходной файл не было записано содержимое. Проверьте структуру входных данных."
            logging.warning(warning_msg)
            print(warning_msg)
            return False
    
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка декодирования JSON: {e}")
        logging.debug(f"Проблемная часть JSON: {e.doc[max(0, e.pos-40):e.pos+40]}")
        print(f"Ошибка: Не удалось разобрать JSON-файл. Проверьте формат файла.")
        return False
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Преобразование JSON-файла в Markdown.")
    parser.add_argument("input_file", help="Путь к входному JSON-файлу")
    parser.add_argument("-o", "--output", help="Путь к выходному MD-файлу", default="output.md")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод отладочной информации")
    parser.add_argument("-e", "--exclude", nargs="+", help="Дополнительные расширения файлов для исключения (например, .js .html)")
    
    args = parser.parse_args()
    
    # Устанавливаем уровень логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Выводим информацию о запуске
    logging.info(f"Запуск скрипта с аргументами: входной файл={args.input_file}, выходной файл={args.output}")
    
    # Определяем список исключаемых расширений
    excluded_extensions = [".css", ".svg", ".scss"]
    if args.exclude:
        excluded_extensions.extend(args.exclude)
    
    # Вызываем функцию преобразования
    result = convert_json_to_md(args.input_file, args.output, excluded_extensions)
    
    # Если преобразование не удалось, выходим с кодом ошибки
    if result is False:
        import sys
        sys.exit(1)
