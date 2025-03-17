"""
Модуль конфигурации для PromptFusion

Содержит классы и функции для управления конфигурацией приложения и проекта.

"""

import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Set

load_dotenv ()

# Константы
DATA_ROOT = './_data'
LARGE_SOURCE_FILE = 40000  # Размер в байтах для определения больших файлов

# Режимы описания
MODE_DESC = "desc"  # Стандартный режим с подробными описаниями
MODE_DESC_NO = "desc_no"  # Без описаний
MODE_DESC_2 = "desc_2"  # Сокращенные описания

# Списки исключений по умолчанию
DEFAULT_EXCLUDED_DIRS = [
    "node_modules/",
    "dist/",
    "build/",
    ".git/",
    "__pycache__/",
    ".venv/",
    "venv/",
    "src/fonts/",
    "src/img/",
    "src/locales/"
]

DEFAULT_EXCLUDED_FILE_TYPES = [
    "woff", "woff2", "eot", "ttf", "otf",  # шрифты
    "svg", "png", "jpg", "jpeg", "gif", "ico",  # изображения
    "po", "mo",  # файлы локализации
    "lock",  # lock-файлы пакетных менеджеров
    "map",  # map-файлы для отладки
    "pyc", "pyo", "pyd"  # байт-код Python
]

DEFAULT_EXCLUDED_FILE_NAMES = [
    "yarn.lock",
    "package-lock.json",
    "messages.po",
    ".DS_Store",
    "Thumbs.db"
]


@dataclass
class FilterConfig:
    """Конфигурация фильтрации файлов проекта"""
    excluded_dirs: List[str] = field (default_factory=list)
    excluded_file_types: List[str] = field (default_factory=list)
    excluded_file_names: List[str] = field (default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Конвертирует конфигурацию в словарь для сериализации"""
        return {
            "excluded_dirs": self.excluded_dirs,
            "excluded_file_types": self.excluded_file_types,
            "excluded_file_names": self.excluded_file_names
        }

    @staticmethod
    def from_dict(data: Dict[str, List[str]]) -> 'FilterConfig':
        """Создает конфигурацию из словаря"""
        return FilterConfig (
            excluded_dirs=data.get ("excluded_dirs", []),
            excluded_file_types=data.get ("excluded_file_types", []),
            excluded_file_names=data.get ("excluded_file_names", [])
        )



class AppConfig(BaseModel):
    """Конфигурация приложения openai/gpt-4o-mini minimax/minimax-01"""
    proj_folder: str = Field('', description="Текущая папка проекта")
    description_model: str = Field('openai/gpt-4o-mini', description="Модель для создания описаний файлов")
    embedding_model_path: str = Field('sentence-transformers/all-MiniLM-L6-v2', description="Модель для создания эмбеддингов")
    chat_model: str = Field('minimax/minimax-01', description="Модель для чата с кодом")
    base_url: str = Field("https://openrouter.ai/api/v1", description="URL API LLM")
    api_key: str = Field("", description="API ключ (будет заменен переменной окружения OPENROUTER_API_KEY)")
    default_project_include: List[str] = Field(['**/*'], description="Шаблоны глоб для включения файлов по умолчанию")
    default_project_exclude: List[str] = Field([], description="Шаблоны глоб для исключения файлов по умолчанию")
    default_project_gitignore: bool = Field(True, description="Учитывать ли .gitignore по умолчанию")
    default_project_remove_comments: bool = Field(False, description="Удалять ли комментарии из кода по умолчанию")
    default_project_desc_mode: str = Field(MODE_DESC, description="Режим описания файлов по умолчанию")
    default_excluded_dirs: List[str] = Field(DEFAULT_EXCLUDED_DIRS, description="Директории, исключаемые по умолчанию")
    default_excluded_file_types: List[str] = Field(DEFAULT_EXCLUDED_FILE_TYPES, description="Типы файлов, исключаемые по умолчанию")
    default_excluded_file_names: List[str] = Field(DEFAULT_EXCLUDED_FILE_NAMES, description="Имена файлов, исключаемые по умолчанию")
    verbose_log: bool = Field(False, description="Подробное логирование")


class ProjConfig(BaseModel):
    """Конфигурация проекта"""
    path: str = Field('', description="Абсолютный путь к папке проекта")
    include: List[str] = Field(['**/*'], description="Шаблоны глоб для включения файлов")
    exclude: List[str] = Field([], description="Шаблоны глоб для исключения файлов")
    gitignore: bool = Field(True, description="Учитывать ли .gitignore")
    remove_comments: bool = Field(False, description="Удалять ли комментарии из кода")
    desc_mode: str = Field(MODE_DESC, description="Режим описания файлов")
    excluded_dirs: List[str] = Field([], description="Директории, исключаемые из анализа")
    excluded_file_types: List[str] = Field([], description="Типы файлов, исключаемые из анализа")
    excluded_file_names: List[str] = Field([], description="Имена файлов, исключаемые из анализа")


def get_app_config_path() -> str:
    """Возвращает путь к файлу конфигурации приложения"""
    return os.path.join (DATA_ROOT, 'app_config.json')


def get_proj_config_path(proj_folder: str) -> str:
    """Возвращает путь к файлу конфигурации проекта"""
    return os.path.join (get_proj_data_folder (proj_folder), 'proj_config.json')


def get_proj_state_path(proj_folder: str) -> str:
    """Возвращает путь к файлу состояния проекта"""
    return os.path.join (get_proj_data_folder (proj_folder), 'proj_state.json')


def get_proj_data_folder(proj_folder: str) -> str:
    """Возвращает путь к папке данных проекта"""
    return os.path.join (DATA_ROOT, proj_folder)


def ensure_data_folders(proj_folder: Optional[str] = None) -> None:
    """Создает необходимые папки для данных"""
    if not os.path.exists (DATA_ROOT):
        os.makedirs (DATA_ROOT)

    if proj_folder:
        proj_data_folder = get_proj_data_folder (proj_folder)
        if not os.path.exists (proj_data_folder):
            os.makedirs (proj_data_folder)


def load_app_config() -> AppConfig:
    """Загружает конфигурацию приложения"""

    config_path = get_app_config_path ()

    # Создаем конфигурацию по умолчанию, если файл не существует
    if not os.path.exists (config_path):
        default_config = AppConfig ()
        # Заменяем API ключ на значение из переменной окружения, если она существует
        if os.environ.get ('OPENROUTER_API_KEY'):
            default_config.api_key = os.environ.get ('OPENROUTER_API_KEY')
            print (f"API key: {default_config.api_key}")
            print (f"Description model: {default_config.description_model}")
            print (f"Chat model: {default_config.chat_model}")
        save_app_config (default_config)
        return default_config

    with open (config_path, 'r', encoding='utf-8') as file:
        config_data = json.load (file)

    config_data = AppConfig ()
    app_config = AppConfig.model_validate (config_data)

    # Заменяем API ключ на значение из переменной окружения, если она существует
    if os.environ.get ('OPENROUTER_API_KEY'):
        app_config.api_key = os.environ.get ('OPENROUTER_API_KEY')
        print(f"API key: {app_config.api_key}")

    print (f"Description model: {app_config.description_model}")
    print (f"Chat model: {app_config.chat_model}")
    return app_config


def save_app_config(app_config: AppConfig) -> None:
    """Сохраняет конфигурацию приложения"""
    ensure_data_folders ()

    with open (get_app_config_path (), 'w', encoding='utf-8') as file:
        json.dump (app_config.model_dump (), file, indent=4)


def load_proj_config(proj_folder: str) -> ProjConfig:
    """Загружает конфигурацию проекта"""
    with open (get_proj_config_path (proj_folder), 'r', encoding='utf-8') as file:
        proj_config_data = json.load (file)

    return ProjConfig.model_validate (proj_config_data)


def save_proj_config(proj_config: ProjConfig, proj_folder: str) -> None:
    """Сохраняет конфигурацию проекта"""
    ensure_data_folders (proj_folder)

    with open (get_proj_config_path (proj_folder), 'w', encoding='utf-8') as file:
        json.dump (proj_config.model_dump (), file, indent=4)


def is_valid_desc_mode(mode: str) -> bool:
    """Проверяет, является ли режим описания допустимым"""
    return mode in [MODE_DESC, MODE_DESC_NO, MODE_DESC_2]