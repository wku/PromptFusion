"""
Модуль моделей данных для PromptFusion

Содержит классы, представляющие состояние приложения, проекта и файлов.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from openai import OpenAI

from config import AppConfig, ProjConfig


class FileNode:
    """
    Представляет файл или папку в файловой системе проекта.
    Используется для построения дерева файлов.
    """

    def __init__(self, name: str, is_folder: bool, folder_content: List = None, size: int = 0, tokens: int = 0):
        self.name = name
        self.is_folder = is_folder
        self.folder_content = folder_content or []
        self.size = size
        self.tokens = tokens

    def __repr__(self):
        return f"FileNode(name={self.name}, is_folder={self.is_folder}, size={self.size}, tokens={self.tokens})"


class FileState (BaseModel):
    """
    Хранит информацию о файле проекта: путь, время модификации,
    описание, эмбеддинги и структурный анализ.
    """
    path: str = ""
    mtime: int = 0
    desc: str = ""
    desc2: str = ""
    embed: List[float] = []
    structure: Optional[Dict[str, Any]] = None


class ProjectStructure (BaseModel):
    """
    Представляет структурный анализ всего проекта
    """
    class_count: int = 0
    function_count: int = 0
    file_types: Dict[str, int] = {}
    external_dependencies: List[str] = []
    internal_dependencies: Dict[str, List[str]] = {}


class ProjStat (BaseModel):
    """
    Статистика проекта по файлам и их размерам
    """
    file_count: int = 0
    total_size: int = 0
    total_tokens: int = 0
    large_files: List[Dict[str, Union[str, int]]] = []

    def __str__(self):
        from utils import bytes_to_str

        large_files_str = ""
        if not self.large_files:
            large_files_str = " Нет больших файлов"
        else:
            large_files_str = "\n".join (
                f" {i + 1}. {f['path']} - Size: {bytes_to_str (f['size'])}"
                for i, f in enumerate (self.large_files)
            )

        return (
            f"Общее количество файлов: {self.file_count}\n"
            f"Общий размер файлов: {bytes_to_str (self.total_size)}\n"
            f"Общее количество токенов: {self.total_tokens}\n"
            f"Большие файлы:\n{large_files_str}"
        )


class ProjState (BaseModel):
    """
    Состояние проекта, включая информацию о файлах и структуре проекта
    """
    remove_comments: bool = False
    files: List[FileState] = []
    structure: Optional[ProjectStructure] = None


class AppState:
    """
    Общее состояние приложения, включающее все остальные состояния
    """

    def __init__(self):
        self.app_config: Optional[AppConfig] = None
        self.proj_config: Optional[ProjConfig] = None
        self.file_paths: List[str] = []
        self.proj_state: Optional[ProjState] = None
        self.proj_stat: Optional[ProjStat] = None
        self.openai: Optional[OpenAI] = None


class ChatSession:
    """
    Представляет сессию чата, включая историю сообщений и учёт токенов/затрат
    """

    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.request_in_tokens: int = 0
        self.request_out_tokens: int = 0
        self.total_in_tokens: int = 0
        self.total_out_tokens: int = 0
        self.exit_requested: bool = False