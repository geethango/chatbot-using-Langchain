@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\geeth\anaconda3\condabin\conda.bat" activate "d:\task2\venv"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@python c:\Users\geeth\.vscode\extensions\ms-python.python-2025.2.0-win32-x64\python_files\get_output_via_markers.py c:/Users/geeth/.vscode/extensions/ms-python.python-2025.2.0-win32-x64/python_files/printEnvVariables.py
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
