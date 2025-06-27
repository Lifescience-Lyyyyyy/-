@echo off
echo =======================================================
echo  正在启动 蚁群算法演示程序 后端服务器...
echo  请勿关闭此窗口。
echo  程序启动后，会自动在您的默认浏览器中打开。
echo  (已启用 --reload 模式，修改py文件会自动重启)
echo =======================================================

REM 进入脚本所在的目录
cd /d "%~dp0"

REM 检查是否安装了Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到Python。请先安装Python并将其添加到系统PATH。
    pause
    exit /b
)

REM 进入后端目录
cd backend
if not exist "main.py" (
    echo 错误: 找不到 backend/main.py。请确保您的目录结构正确。
    pause
    exit /b
)

REM 启动服务器。使用 --reload 选项，这样保存py文件时服务器会自动重启。
echo 正在启动Uvicorn服务器...
start "ACO Server" uvicorn main:app --host 127.0.0.1 --port 8000 --reload

REM 等待几秒钟，确保服务器有足够的时间启动
echo 等待服务器初始化...
timeout /t 3 /nobreak > nul

REM 在浏览器中打开应用
echo 在浏览器中打开 http://127.0.0.1:8000
start "" http://127.0.0.1:8000

echo.
echo 服务器正在运行。你可以继续使用本程序。
echo 如需关闭服务器，请关闭新弹出的 "ACO Server" 命令行窗口。

pause