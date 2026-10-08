"""Bounded retries around the unchanged JSON format and atomic writer."""
import time
from tools.autolab_cem import write as atomic_write


def write(path,value):
    for attempt in range(10):
        try:
            return atomic_write(path,value)
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==9:
                raise
            time.sleep(min(.02*2**attempt,.25))
