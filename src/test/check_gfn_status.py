import sys
import gfn
import importlib.util

print(f"Python path: {sys.executable}")
print(f"gfn path: {gfn.__file__}")
print(f"gfn paths: {getattr(gfn, '__path__', None)}")
print(f"gfn.constants found: {importlib.util.find_spec('gfn.constants') is not None}")
