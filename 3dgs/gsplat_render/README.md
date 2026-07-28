# build  
## Windows  
```powershell
. "E:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
$env:DISTUTILS_USE_SDK = "1"
$env:CUDA_HOME = "D:\Program Files\NVIDIA\CUDA\v13.2"
$env:CUDA_PATH = "D:\Program Files\NVIDIA\CUDA\v13.2"
$env:Path = "D:\Program Files\NVIDIA\CUDA\v13.2\bin;$env:Path"
cd /path/to/.../mymuladd
python -m pip install -v -e . --no-build-isolation
```