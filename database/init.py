import sys
import django
import os
from pathlib import Path

def init(cwd: str = ""):
    cwd = cwd or str(Path(os.getcwd()) / "data")
    
    sys.path.append(cwd)
    # sys.path.pop(0)
    # sys.path.append('')
    print(sys.path)
    print(cwd)
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
    django.setup()
