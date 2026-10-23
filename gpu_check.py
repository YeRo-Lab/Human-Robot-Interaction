# Standalone Whisper speed check, CPU vs GPU. Run it on its own, NOT through server.py:
# a missing CUDA library can kill the process outright, and this keeps that away from the server.
#
#   python gpu_check.py                          (uses 3 s of silence)
#   python gpu_check.py audio\1captured_audio.wav (uses a real recording, better)
#   python gpu_check.py audio\1captured_audio.wav base small large-v3-turbo
import sys
import time

import numpy as np

import gpu_libs
gpu_libs.add_nvidia_libs_to_path()

import ctranslate2
from faster_whisper import WhisperModel

print("ctranslate2", ctranslate2.__version__, "| CUDA devices:", ctranslate2.get_cuda_device_count(), flush=True)

source = sys.argv[1] if len(sys.argv) > 1 else np.zeros(16000 * 3, dtype=np.float32)
models = sys.argv[2:] or ["base", "small"]

for device, compute in (("cpu", "int8"), ("cuda", "float16")):
    for name in models:
        label = "%s on %s (%s)" % (name, device, compute)
        try:
            t0 = time.time()
            model = WhisperModel(name, device=device, compute_type=compute)
            load = time.time() - t0
            list(model.transcribe(source)[0])            # warm-up (first call is slow)
            t1 = time.time()
            segments, info = model.transcribe(source)
            text = " ".join(s.text for s in segments).strip()   # transcribe() is lazy, this runs it
            took = time.time() - t1
            print("%-38s load %5.1fs | transcribe %5.2fs | %r" % (label, load, took, text[:70]), flush=True)
        except Exception as e:
            print("%-38s FAILED: %s" % (label, e), flush=True)
