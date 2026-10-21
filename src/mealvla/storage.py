"""UTF-8 configuration and atomic run metadata writes."""

import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            name = stream.name
            stream.write(payload + "\n")
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)
