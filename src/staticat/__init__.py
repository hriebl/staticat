from pathlib import Path

import pydantic

from .lib import Catalog


class ConfigTOML(pydantic.BaseModel):
    directory: Path
    catalog_template: Path | None = None
    dataset_template: Path | None = None
    convert_excel: bool = True


def staticat(config):
    catalog = Catalog(config)
    catalog.process()
