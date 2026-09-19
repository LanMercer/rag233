@echo off
rem ============================================================
rem 第四周 · 一键启动（Day17 雏形）——双击我，或在终端里 start.bat
rem   等价于：python start.py
rem   想只体检：start.bat --check
rem   想自动起模型服务：start.bat --start-service
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem 优先用 llm 环境里的 python；找不到就退回 PATH 里的 python
set PY=D:\miniconda1\envs\llm\python.exe
if not exist "%PY%" set PY=python

"%PY%" start.py %*
echo.
echo [结束] 按任意键关闭本窗口……
pause >nul
