# Preload pip-installed cuDNN libraries before ctranslate2 (whisperX dependency)
# loads its bundled cuDNN 9.1.0 which expects matching sub-libraries.
# Without this, ctranslate2 finds version-mismatched system cuDNN and crashes.
import ctypes
import glob
import os

try:
    import nvidia.cudnn
    _cudnn_dir = os.path.join(nvidia.cudnn.__path__[0], "lib")
    for _lib in sorted(glob.glob(os.path.join(_cudnn_dir, "libcudnn*.so.9"))):
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
except (ImportError, OSError, IndexError):
    pass

from app.main import app  # noqa: F401

# This is the entry point for uvicorn
# Run with: uvicorn main:app --reload
