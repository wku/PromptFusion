"""
Модуль для реализации сессии чата с использованием промпт-инструментов
вместо механизма функций API OpenAI/OpenRouter.


"""

import os
import json
from typing import List, Dict, Any, Optional, Union

from models import AppState, ChatSession
from utils import get_tokens_cnt, get_cost, input_yes_no, is_no, is_yes
from prompt_tools import PromptTools


def start_chat_session(app_state: AppState) -> None:
    """
    Запускает интерактивный чат с LLM для обсуждения проекта
    с использованием промпт-инструментов вместо функций API.

    Args:
        app_state: состояние приложения
    """
    session = ChatSession ()
    prompt_tools = PromptTools (app_state)

    # Добавляем системное сообщение
    sys_prompt = get_sys_prompt (app_state)

    # Расширяем системный промпт инструкциями по работе с инструментами
    enhanced_sys_prompt = prompt_tools.enhance_system_prompt (sys_prompt)

    sys_prompt_tokens = get_tokens_cnt (enhanced_sys_prompt)
    sys_prompt_cost = get_cost (app_state.app_config.chat_model, sys_prompt_tokens, 0)

    print (f"\nСистемный промпт: стоимость ввода: ${sys_prompt_cost:.4f} ({sys_prompt_tokens} токенов)")
    print (f"\nТеперь вы можете задавать вопросы о проекте {os.path.basename (app_state.proj_config.path)}.")
    print ("Нажмите Ctrl-C для прерывания в любой момент.")
    print ("Используйте команду /exit для выхода или /clear для очистки истории.")

    # Инициализируем сессию
    session.messages.append ({'role': 'system', 'content': enhanced_sys_prompt})

    while not session.exit_requested:
        try:
            # Получаем ввод пользователя
            user_input = get_user_input (session)
            if session.exit_requested:
                break

            # Проверяем на наличие вызова функции в вводе пользователя
            function_result = prompt_tools.process_user_message (user_input)
            if function_result:
                # Если пользователь ввел вызов функции напрямую, обрабатываем его
                formatted_result = prompt_tools.format_function_result (function_result)
                print (f"\n{formatted_result}")
                continue

            # Добавляем сообщение пользователя в историю
            session.messages.append ({'role': 'user', 'content': user_input})

            # Отправляем запрос к LLM
            print ("\nОжидание ответа модели...")

            try:
                response = app_state.openai.chat.completions.create (
                    model=app_state.app_config.chat_model,
                    messages=session.messages,
                    temperature=0
                )

                # Для отладки - выводим полный ответ API
                print ("\nAPI Response:", response)

            except Exception as e:
                print (f"Ошибка при запросе к API: {e}")
                continue

            # Проверяем на возможные проблемы с API
            if not hasattr (response, 'usage') or response.usage is None:
                print (f"Usage: app_state.app_config.chat_model {app_state.app_config.chat_model}")
                print ("Ошибка: response.usage вернул None. Возможные причины: неудачный API-запрос или проблемы с авторизацией.")
                continue

            # Обновляем статистику токенов
            session.request_in_tokens = response.usage.prompt_tokens
            session.request_out_tokens = response.usage.completion_tokens
            session.total_in_tokens += response.usage.prompt_tokens
            session.total_out_tokens += response.usage.completion_tokens

            # Получаем ответ модели
            response_message = response.choices[0].message
            bot_response = response_message.content

            # Обрабатываем ответ бота, заменяя вызовы функций на их результаты
            processed_response = bot_response

            while True:
                original_response = processed_response
                processed_response = prompt_tools.process_bot_response (processed_response)

                # Если ответ не изменился после обработки, значит все функции обработаны
                if original_response == processed_response:
                    break

            # Добавляем обработанный ответ в историю сообщений
            session.messages.append ({'role': 'assistant', 'content': processed_response})

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
            print (f'🤖 Бот: {processed_response}')

            # Сбрасываем счетчики для следующего запроса
            session.request_in_tokens = 0
            session.request_out_tokens = 0

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
            # Сохраняем системное сообщение
            system_message = next ((msg for msg in session.messages if msg['role'] == 'system'), None)
            session.messages = []
            if system_message:
                session.messages.append (system_message)
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


def get_sys_prompt(app_state: AppState) -> str:
    """
    Формирует системный промпт для чата в зависимости от режима описания.

    Args:
        app_state: состояние приложения

    Returns:
        str: системный промпт
    """
    # Импортируем только при необходимости для избежания циклических импортов
    from config import MODE_DESC, MODE_DESC_NO, MODE_DESC_2

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


def get_sys_prompt_template(proj_struct: str, project_structure: Any, is_desc: bool) -> str:
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

Файлы проекта{with_descs}:

{proj_struct}
"""


def get_sys_context_desc(files: List[Any]) -> str:
    """
    Формирует контекст с полными описаниями файлов

    Args:
        files: список файлов

    Returns:
        str: форматированный список файлов с описаниями
    """
    from utils import add_path_prefix
    return '\n\n'.join (f"{add_path_prefix (file.path)}\n{file.desc}" for file in files)


def get_sys_context_no_desc(files: List[Any]) -> str:
    """
    Формирует контекст без описаний файлов

    Args:
        files: список файлов

    Returns:
        str: форматированный список файлов
    """
    from utils import add_path_prefix
    return '\n'.join (f"{add_path_prefix (file.path)}" for file in files)


def get_sys_context_short_desc(files: List[Any]) -> str:
    """
    Формирует контекст с краткими описаниями файлов

    Args:
        files: список файлов

    Returns:
        str: форматированный список файлов с краткими описаниями
    """
    from utils import add_path_prefix
    return '\n\n'.join (f"{add_path_prefix (file.path)}\n{file.desc2}" for file in files)



