# Timestamped logging for the client. Same output format as the old
# MemoryListenerModule._log: "[HH:MM:SS.mmm] message".
from __future__ import print_function
import time


def _ts():
    millis = int((time.time() % 1) * 1000)
    return time.strftime("%H:%M:%S", time.localtime()) + ".%03d" % millis


def log(*args):
    message = " ".join(str(a) for a in args)
    print("[" + _ts() + "] " + message)
