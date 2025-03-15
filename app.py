#!/usr/bin/env python3
"""
PromptFusion: инструмент для взаимодействия с кодовыми базами через LLM

Основной модуль, который служит точкой входа в приложение.
Координирует ключевые этапы работы:
1. Инициализация состояния приложения и проекта
2. Анализ и создание описаний файлов проекта
3. Интерактивный чат с моделью LLM для обсуждения кодовой базы
"""

import sys
from dotenv import load_dotenv
from openai import OpenAI

from models import AppState
from config import AppConfig, load_app_config, save_app_config

#from core import initialize_project, analyze_project_files, start_chat_session
from core import initialize_project, analyze_project_files
from improved_chat_session import start_chat_session

from utils import setup_logging


def main():
    """
    Основная функция приложения, координирующая рабочий процесс.
    """
    # Загружаем переменные окружения
    load_dotenv ()

    # Настраиваем логирование
    logger = setup_logging ()
    logger.info ("Запуск PromptFusion")

    try:
        # Инициализируем состояние приложения
        app_state = AppState ()

        # Загружаем конфигурацию приложения
        app_config = load_app_config ()
        app_state.app_config = app_config

        # Инициализируем проект
        logger.info ("Инициализация проекта")
        if not initialize_project (app_state):
            logger.info ("Инициализация прервана пользователем")
            return

        # Инициализируем клиент OpenAI
        app_state.openai = OpenAI (
            base_url=app_state.app_config.base_url,
            api_key=app_state.app_config.api_key
        )

        # Анализируем файлы проекта и создаем их описания
        logger.info ("Анализ файлов проекта")
        if not analyze_project_files (app_state):
            logger.info ("Анализ прерван пользователем")
            return

        # Запускаем интерактивный чат
        logger.info ("Запуск интерактивного чата")
        start_chat_session (app_state)

    except KeyboardInterrupt:
        print ("\nПриложение прервано пользователем")
    except Exception as e:
        print (f"\nОшибка: {e}")
        logger.exception ("Неожиданная ошибка:")
    finally:
        print ("\nЗавершение работы PromptFusion")


if __name__ == "__main__":
    main ()