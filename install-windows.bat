@echo off

REM Create a new virtual environment
echo Creating virtual environment...
python -m venv myenv

REM Activate the virtual environment
echo Activating virtual environment...
call myenv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Deactivate the virtual environment
echo Deactivating virtual environment...
call deactivate.bat

echo Done.
pause