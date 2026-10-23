# Makes pip-installed NVIDIA runtime DLLs (cuBLAS / cuDNN) findable on Windows.
import importlib
import os


def add_nvidia_libs_to_path():
    if os.name != "nt":
        return
    for pkg in ("cublas", "cudnn"):
        try:
            mod = importlib.import_module("nvidia." + pkg)
        except ImportError:
            continue
        bin_dir = os.path.join(list(mod.__path__)[0], "bin")
        if os.path.isdir(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
