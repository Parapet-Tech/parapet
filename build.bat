@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\Users\anyth\MINE\dev\parapet\parapet
cargo +stable-x86_64-pc-windows-msvc build --release
