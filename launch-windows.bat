@echo off

REM Activate the virtual environment
echo Activating virtual environment...
call myenv\Scripts\activate.bat

REM Start the Python application
echo Starting Python application...
streamlit run Home.py

REM Deactivate the virtual environment
echo Deactivating virtual environment...
call deactivate.bat

echo Done.
pause