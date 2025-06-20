from pathlib import Path
from typing import Optional

import pydantic

from .lib import Catalog


class ConfigTOML(pydantic.BaseModel):
    directory: Path
    catalog_template: Optional[Path] = None
    dataset_template: Optional[Path] = None
    convert_excel: bool = True
    ignore: list[str] = [".*"]


def staticat(config):
    catalog = Catalog(config)
    catalog.process()
