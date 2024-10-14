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


def urlname(value):
    return PurePosixPath(unquote(urlparse(value).path)).name


def jinja_env(loader):
    autoescape = jinja2.select_autoescape(("html", "htm", "xml", "rdf"))
    env = jinja2.Environment(loader=loader, autoescape=autoescape)
    env.filters["urlname"] = urlname
    return env


def default_template(name):
    env = jinja_env(jinja2.PackageLoader("staticat"))
    return env.get_template(name)


def custom_template(path):
    env = jinja_env(jinja2.FileSystemLoader(path.parent))
    return env.get_template(path.name)


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
        staticat_config = catalog.staticat_config
        log_directory = directory.relative_to(staticat_config.directory.parent)
        logger.info(f"{log_directory}: Parsing dataset.toml")

        try:
            with open(directory / "dataset.toml", "rb") as file:
                kwargs = catalog.dataset_defaults | tomllib.load(file)
                super().__init__(**kwargs)

                self.political_geocoding_level
        except Exception as error:
            raise Exception("Could not parse dataset.toml") from error

        self._directory = directory
        self._staticat_config = staticat_config
        self._catalog_uri = catalog.uri

    @property
    def catalog_uri(self):
        return self._catalog_uri

    @property
    def directory(self):
        return self._directory

    @property
    def html_description(self):
        return MarkdownIt("js-default").render(self.description)

    @property
    def html_template(self):
        if self.staticat_config.dataset_template is None:
            return default_template("dataset.html")

        return custom_template(self.staticat_config.dataset_template)

    @property
    def log_directory(self):
        return self.directory.relative_to(self.staticat_config.directory.parent)

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
        return self.directory.relative_to(self.staticat_config.directory)

    @property
    def should_convert_excel(self):
        if self.config.convert_excel is None:
            return self.staticat_config.convert_excel

        return self.config.convert_excel

    @property
    def staticat_config(self):
        return self._staticat_config

    @property
    def uri(self):
        return f"{self.catalog_uri}/{quote(self.relative_directory.as_posix())}"

    def add_distributions(self):
        for file in self.directory.glob("*"):
            if not file.is_file():
                continue

            if file.name in ("dataset.toml", "index.html"):
                continue

            if self.should_convert_excel and file.suffix in (".xls", ".xlsx"):
                continue

            if file.suffix not in FileTypeDF.index:
                logger.warning(
                    f"{self.log_directory}: "
                    f"Skipping {file.name}: "
                    "File type not supported"
                )

                continue

            logger.info(f"{self.log_directory}: Adding {file.name}")

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

    def convert_excel(self):
        for file in self.directory.glob("*"):
            if not file.is_file():
                continue

            if file.suffix not in (".xls", ".xlsx"):
                continue

            logger.info(f"{self.log_directory}: Converting {file.name}")

            try:
                df = pd.read_excel(file)
                csv = self.directory / f"{file.stem}.csv"
                df.to_csv(csv, index=False)

                os.utime(csv, (file.stat().st_atime, file.stat().st_mtime))
            except Exception as error:
                logger.error(
                    f"{self.log_directory}: "
                    f"Could not convert {file.name}: "
                    f"{error}"
                )

    def write_html(self):
        logger.info(f"{self.log_directory}: Writing index.html")

        try:
            with open(self.directory / "index.html", "w", encoding="utf-8") as file:
                file.write(self.html_template.render(dataset=self))
        except Exception as error:
            raise Exception("Could not write index.html") from error

    def process(self):
        if self.should_convert_excel:
            self.convert_excel()

        self.add_distributions()
        self.write_html()


class Catalog(CatalogTOML):
    def __init__(self, config):
        logger.info(f"{config.directory.name}: Parsing catalog.toml")

        try:
            with open(config.directory / "catalog.toml", "rb") as file:
                super().__init__(**tomllib.load(file))
        except Exception as error:
            raise Exception("Could not parse catalog.toml") from error

        self._staticat_config = config
        self._datasets = []
        self._tree = []

    @property
    def datasets(self):
        return self._datasets

    @property
    def directory(self):
        return self.staticat_config.directory

    @property
    def html_description(self):
        return MarkdownIt("js-default").render(self.description)

    @property
    def html_template(self):
        if self.staticat_config.catalog_template is None:
            return default_template("catalog.html")

        return custom_template(self.staticat_config.catalog_template)

    @property
    def staticat_config(self):
        return self._staticat_config

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, value):
        self._tree = value

    def build_tree(self):
        datasets = {dataset.relative_directory for dataset in self.datasets}
        parents = {parent for dataset in datasets for parent in dataset.parents}
        items = sorted((datasets | parents) - {Path(".")})

        self.tree = [
            {
                "name": item.name,
                "href": quote((item / "index.html").as_posix()),
                "class": "dataset" if item in datasets else "directory",
                "depth": len(item.parents) - 1,
            }
            for item in items
        ]

    def write_css(self):
        logger.info(f"{self.directory.name}: Writing default.css")

        try:
            with open(self.directory / "default.css", "w", encoding="utf-8") as file:
                file.write(default_template("default.css").render())
        except Exception as error:
            raise Exception("Could not write default.css") from error

    def write_html(self):
        logger.info(f"{self.directory.name}: Writing index.html")

        try:
            with open(self.directory / "index.html", "w", encoding="utf-8") as file:
                file.write(self.html_template.render(catalog=self))
        except Exception as error:
            raise Exception("Could not write index.html") from error

    def write_ttl(self):
        logger.info(f"{self.directory.name}: Writing catalog.ttl")

        try:
            template = default_template("catalog.rdf")

            graph = Graph()
            graph.parse(format="xml", data=template.render(catalog=self))
            graph.serialize(self.directory / "catalog.ttl", encoding="utf-8")
        except Exception as error:
            raise Exception("Could not write catalog.ttl") from error

    def process(self):
        logger.info(f"{self.directory.name}: Processing catalog...")

        for file in self.directory.glob("*/**/dataset.toml"):
            if not file.is_file():
                continue

            log_directory = file.parent.relative_to(self.directory.parent)
            logger.info(f"{log_directory}: Adding dataset...")

            try:
                dataset = Dataset(file.parent, catalog=self)
                dataset.process()

                self.datasets.append(dataset)
            except Exception as error:
                logger.error(
                    f"{log_directory}: Could not add dataset: {error}"
                    + (f": {error.__cause__}" if error.__cause__ else "")
                )

        try:
            self.build_tree()
            self.write_ttl()
            self.write_css()
            self.write_html()
        except Exception as error:
            logger.critical(
                f"{log_directory}: Could not process catalog: {error}"
                + (f": {error.__cause__}" if error.__cause__ else "")
            )

            raise Exception("Could not process catalog") from error
