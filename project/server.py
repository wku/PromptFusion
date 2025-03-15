import asyncio
from aiohttp import web


# Определение асинхронного обработчика
async def handle_hello(request):
    return web.Response (text="Hello World")


# Функция для создания и настройки приложения
async def init_app():
    app = web.Application ()

    # Настройка маршрутов
    app.add_routes ([
        web.get ('/', handle_hello),  # Корневой маршрут
        web.get ('/hello', handle_hello)  # Дополнительный маршрут
    ])

    return app


# Запуск приложения
if __name__ == '__main__':
    loop = asyncio.get_event_loop ()
    app = loop.run_until_complete (init_app ())

    # Настройка параметров запуска
    web.run_app (
        app,
        host='0.0.0.0',  # Слушать все входящие соединения
        port=8080  # Порт для веб-сервиса
    )