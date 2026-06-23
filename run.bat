@echo off
:: Переходим в папку, где лежит сам батник
cd /d "%~dp0"

:: Проверяем, существует ли виртуальное окружение, если нет — предлагаем создать
if not exist ".venv" (
    echo Virtual environment not found! Please create it first.
    pause
    exit /b
)

:: Активация и запуск
call .venv\Scripts\activate
start /B streamlit run gui.py --server.headless true
timeout /t 5
start http://localhost:8501