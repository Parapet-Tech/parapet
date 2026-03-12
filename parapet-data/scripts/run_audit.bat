@echo off
cd /d C:\Users\anyth\MINE\dev\DefenseSector\parapet\parapet-data
python scripts\audit_curated.py > scripts\audit_output.txt 2>&1
echo Done. Exit code: %ERRORLEVEL%
