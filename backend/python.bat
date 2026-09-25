@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%python\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: "%PYTHON_EXE%" not found.
    exit /b 1
)
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONDONTWRITEBYTECODE=1"
set "TEMP=%ROOT%.cache\tmp"
set "TMP=%TEMP%"
set "PIP_CACHE_DIR=%ROOT%.cache\pip"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "HF_HOME=%ROOT%.cache\huggingface"
set "HF_HUB_OFFLINE=1"
set "MPLCONFIGDIR=%ROOT%.cache\matplotlib"
if not exist "%TEMP%" mkdir "%TEMP%"
rem sysconfig already targets this embedded runtime. Do not append --target:
rem target installs can leave multiple metadata versions and overwrite packages.
rem For multiline Python use a script file instead of a multiline -c argument.
"%PYTHON_EXE%" %*
exit /b %ERRORLEVEL%

