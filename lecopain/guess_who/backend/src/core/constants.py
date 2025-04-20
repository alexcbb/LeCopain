from pathlib import Path

from fastapi.templating import Jinja2Templates


BASE_PATH = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))
