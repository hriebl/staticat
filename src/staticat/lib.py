import logging
import os
import tomllib
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

import jinja2
import pandas as pd
import pydantic
from markdown_it import MarkdownIt
from rdflib import Graph

from .vocab import Availability, DataTheme, FileType, FileTypeDF, License

logger = logging.getLogger(__name__)

jinja2_env = jinja2.Environment(
    loader=jinja2.PackageLoader("staticat"),
    autoescape=jinja2.select_autoescape(("html", "htm", "xml", "rdf")),
)


def urlname(value):
    return PurePosixPath(unquote(urlparse(value).path)).name


jinja2_env.filters["urlname"] = urlname


class ContactTOML(pydantic.BaseModel):
    name: str
    email: str


class PublisherTOML(pydantic.BaseModel):
    name: str
    uri: str


class DistributionTOML(pydantic.BaseModel):
    uri: str
    title: str
    modified: datetime | None = None
    format: FileType | None = None
    media_type: str | None = None
    byte_size: float | None = None
    local: bool = False


class DatasetConfigTOML(pydantic.BaseModel):
    convert_excel: bool | None = None


class DatasetTOML(pydantic.BaseModel):
    title: str
    description: str
    keywords: list[str]
    themes: list[DataTheme]
    issued: datetime
    start_date: datetime
    end_date: datetime
    license: License
    availability: Availability
    spatial: str
    political_geocoding: str
    maintainer: ContactTOML
    creator: ContactTOML
    publisher: PublisherTOML
    distributions: list[DistributionTOML] = []
    config: DatasetConfigTOML = DatasetConfigTOML()


class CatalogTOML(pydantic.BaseModel):
    uri: str
    title: str
    description: str
    publisher: PublisherTOML
    dataset_defaults: dict = {}


class Dataset(DatasetTOML):
    def __init__(self, directory, catalog):
        log_directory = directory.relative_to(catalog.config.directory.parent)
        logger.info(f"{log_directory}: Parsing dataset.toml")

        try:
            with open(directory / "dataset.toml", "rb") as file:
                kwargs = catalog.dataset_defaults | tomllib.load(file)
                super().__init__(**kwargs)

                self.political_geocoding_level
        except Exception as error:
            raise Exception("Could not parse dataset.toml") from error

        self._directory = directory
        self._staticat_config = catalog.config
        self._catalog_uri = catalog.uri

    @property
    def _should_convert_excel(self):
        if self.config.convert_excel is None:
            return self._staticat_config.convert_excel

        return self.config.convert_excel

    @property
    def _dataset_template(self):
        return self._staticat_config.dataset_template

    @property
    def _log_directory(self):
        return self._directory.relative_to(self._staticat_config.directory.parent)

    def _add_distributions(self):
        for file in self._directory.glob("*"):
            if not file.is_file():
                continue

            if file.name in ("dataset.toml", "index.html"):
                continue

            if self._should_convert_excel and file.suffix in (".xls", ".xlsx"):
                continue

            if file.suffix not in FileTypeDF.index:
                logger.warning(
                    f"{self._log_directory}: "
                    f"Skipping {file.name}: "
                    "File type not supported"
                )

                continue

            logger.info(f"{self._log_directory}: Adding {file.name}")

            distribution = DistributionTOML(
                title=file.name,
                uri=f"{self.uri}/{quote(file.name)}",
                modified=datetime.fromtimestamp(file.stat().st_mtime),
                format=FileTypeDF.loc[file.suffix]["code"],
                media_type=FileTypeDF.loc[file.suffix]["type"],
                byte_size=file.stat().st_size,
                local=True,
            )

            self.distributions.append(distribution)

    def _convert_excel(self):
        for file in self._directory.glob("*"):
            if not file.is_file():
                continue

            if file.suffix not in (".xls", ".xlsx"):
                continue

            logger.info(f"{self._log_directory}: Converting {file.name}")

            try:
                df = pd.read_excel(file)
                csv = self._directory / f"{file.stem}.csv"
                df.to_csv(csv, index=False)

                os.utime(csv, (file.stat().st_atime, file.stat().st_mtime))
            except Exception as error:
                logger.error(
                    f"{self._log_directory}: "
                    f"Could not convert {file.name}: "
                    f"{error}"
                )

    def _write_html(self):
        logger.info(f"{self._log_directory}: Writing index.html")

        try:
            if self._dataset_template:
                with open(self._dataset_template, "r") as file:
                    template = jinja2.Template(file.read())
            else:
                template = jinja2_env.get_template("dataset.html")

            with open(self._directory / "index.html", "w") as file:
                file.write(template.render(dataset=self))
        except Exception as error:
            raise Exception("Could not write index.html") from error

    @property
    def description_html(self):
        return MarkdownIt("js-default").render(self.description)

    @property
    def political_geocoding_level(self):
        base = "dcat-ap.de/def/politicalGeocoding"

        mapping = {
            "districtKey": "administrativeDistrict",
            "governmentDistrictKey": "administrativeDistrict",
            "municipalAssociationKey": "municipality",
            "municipalityKey": "municipality",
            "regionalKey": "municipality",
            "stateKey": "state",
        }

        for key, value in mapping.items():
            if f"{base}/{key}" in self.political_geocoding:
                return f"http://{base}/Level/{value}"

        raise ValueError("Invalid political geocoding")

    @property
    def relative_catalog(self):
        path = Path(*(".." for parent in self.relative_directory.parents))
        return quote(path.as_posix())

    @property
    def relative_directory(self):
        return self._directory.relative_to(self._staticat_config.directory)

    @property
    def uri(self):
        return f"{self._catalog_uri}/{quote(self.relative_directory.as_posix())}"

    def process(self):
        if self._should_convert_excel:
            self._convert_excel()

        self._add_distributions()
        self._write_html()


class Catalog(CatalogTOML):
    def __init__(self, config):
        logger.info(f"{config.directory.name}: Parsing catalog.toml")

        try:
            with open(config.directory / "catalog.toml", "rb") as file:
                super().__init__(**tomllib.load(file))
        except Exception as error:
            raise Exception("Could not parse catalog.toml") from error

        self._config = config
        self._datasets = []
        self._tree = []

    @property
    def _catalog_template(self):
        return self._config.catalog_template

    @property
    def _directory(self):
        return self._config.directory

    def _build_tree(self):
        datasets = {dataset.relative_directory for dataset in self._datasets}
        parents = {parent for dataset in datasets for parent in dataset.parents}
        items = sorted((datasets | parents) - {Path(".")})

        self._tree = [
            {
                "name": item.name,
                "href": quote((item / "index.html").as_posix()),
                "class": "dataset" if item in datasets else "directory",
                "depth": len(item.parents) - 1,
            }
            for item in items
        ]

    def _write_css(self):
        logger.info(f"{self._directory.name}: Writing default.css")

        try:
            template = jinja2_env.get_template("default.css")

            with open(self._directory / "default.css", "w") as file:
                file.write(template.render())
        except Exception as error:
            raise Exception("Could not write default.css") from error

    def _write_html(self):
        logger.info(f"{self._directory.name}: Writing index.html")

        try:
            if self._catalog_template:
                with open(self._catalog_template, "r") as file:
                    template = jinja2.Template(file.read())
            else:
                template = jinja2_env.get_template("catalog.html")

            with open(self._directory / "index.html", "w") as file:
                file.write(template.render(catalog=self))
        except Exception as error:
            raise Exception("Could not write index.html") from error

    def _write_ttl(self):
        logger.info(f"{self._directory.name}: Writing catalog.ttl")

        try:
            template = jinja2_env.get_template("catalog.rdf")

            graph = Graph()
            graph.parse(format="xml", data=template.render(catalog=self))
            graph.serialize(self._directory / "catalog.ttl")
        except Exception as error:
            raise Exception("Could not write catalog.ttl") from error

    @property
    def config(self):
        return self._config

    @property
    def datasets(self):
        return self._datasets

    @property
    def description_html(self):
        return MarkdownIt("js-default").render(self.description)

    @property
    def tree(self):
        return self._tree

    def process(self):
        logger.info(f"{self._directory.name}: Processing catalog...")

        for file in self._directory.glob("*/**/dataset.toml"):
            if not file.is_file():
                continue

            log_directory = file.parent.relative_to(self._directory.parent)
            logger.info(f"{log_directory}: Adding dataset...")

            try:
                dataset = Dataset(file.parent, catalog=self)
                dataset.process()

                self._datasets.append(dataset)
            except Exception as error:
                logger.error(
                    f"{log_directory}: Could not add dataset: {error}"
                    + (f": {error.__cause__}" if error.__cause__ else "")
                )

        try:
            self._build_tree()
            self._write_ttl()
            self._write_css()
            self._write_html()
        except Exception as error:
            logger.critical(
                f"{log_directory}: Could not process catalog: {error}"
                + (f": {error.__cause__}" if error.__cause__ else "")
            )

            raise Exception("Could not process catalog") from error
