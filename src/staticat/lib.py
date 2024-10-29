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
    """Extracts the last component of a URL path."""
    return PurePosixPath(unquote(urlparse(value).path)).name


def jinja_env(loader):
    """Returns a Jinja environment with autoescape and the urlname filter."""
    autoescape = jinja2.select_autoescape(("html", "htm", "xml", "rdf"))
    env = jinja2.Environment(loader=loader, autoescape=autoescape)
    env.filters["urlname"] = urlname
    return env


def default_template(name):
    """Returns a default Jinja template from the Staticat package."""
    env = jinja_env(jinja2.PackageLoader("staticat", encoding="utf-8"))
    return env.get_template(name)


def custom_template(path):
    """Returns a custom, user-defined Jinja template from the file system."""
    env = jinja_env(jinja2.FileSystemLoader(path.parent, encoding="utf-8"))
    return env.get_template(path.name)


def write(path, data):
    """Writes a file in text mode with UTF-8 encoding."""
    with open(path, mode="w", encoding="utf-8") as file:
        file.write(data)


class ContactTOML(pydantic.BaseModel):
    """TOML configuration of a contact point (DCAT property)."""

    name: str
    email: str


class PublisherTOML(pydantic.BaseModel):
    """TOML configuration of a publisher (DCT property)."""

    name: str
    uri: str


class DistributionTOML(pydantic.BaseModel):
    """TOML configuration of a distribution (DCAT class)."""

    uri: str
    title: str
    modified: datetime | None = None
    format: FileType | None = None
    media_type: str | None = None
    byte_size: float | None = None
    local: bool = False


class DatasetConfigTOML(pydantic.BaseModel):
    """TOML configuration defining Staticat-specific options for a dataset."""

    convert_excel: bool | None = None


class DatasetTOML(pydantic.BaseModel):
    """TOML configuration of a dataset (DCAT class)."""

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
    """TOML configuration of a catalog (DCAT class)."""

    uri: str
    title: str
    description: str
    publisher: PublisherTOML
    dataset_defaults: dict = {}


class Dataset(DatasetTOML):
    """A dataset with RDF properties, Staticat properties and processing methods."""

    def __init__(self, directory, catalog):
        """Initializes the dataset, parsing and validating the file dataset.toml."""
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
        """The URI of the catalog."""
        return self._catalog_uri

    @property
    def directory(self):
        """The directory of the dataset."""
        return self._directory

    @property
    def html_description(self):
        """The description of the dataset, rendered from Markdown to HTML."""
        return MarkdownIt("js-default").render(self.description)

    @property
    def log_directory(self):
        """The directory of the dataset, formatted for logging."""
        return self.directory.relative_to(self.staticat_config.directory.parent)

    @property
    def political_geocoding_level(self):
        """The political geocoding level (DCAT-AP.de property).

        Inferred from the political geocoding given in the TOML configuration.
        """
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
        """The directory of the catalog relative to the dataset."""
        path = Path(*(".." for parent in self.relative_directory.parents))
        return quote(path.as_posix())

    @property
    def relative_directory(self):
        """The directory of the dataset relative to the catalog."""
        return self.directory.relative_to(self.staticat_config.directory)

    @property
    def should_convert_excel(self):
        """Whether Excel files should be converted to CSV."""
        if self.config.convert_excel is None:
            return self.staticat_config.convert_excel

        return self.config.convert_excel

    @property
    def staticat_config(self):
        """The global Staticat configuration."""
        return self._staticat_config

    @property
    def uri(self):
        """The URI of the dataset."""
        return f"{self.catalog_uri}/{quote(self.relative_directory.as_posix())}"

    def add_distributions(self):
        """Adds local files to the dataset as distributions.

        Unsupported file types are skipped.
        """
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

            self.distributions.append(
                DistributionTOML(
                    title=file.name,
                    uri=f"{self.uri}/{quote(file.name)}",
                    modified=datetime.fromtimestamp(file.stat().st_mtime),
                    format=FileTypeDF.loc[file.suffix]["code"],
                    media_type=FileTypeDF.loc[file.suffix]["type"],
                    byte_size=file.stat().st_size,
                    local=True,
                )
            )

    def convert_excel(self):
        """Converts Excel files to CSV."""
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

    def render_html(self):
        """Renders the website of the dataset."""
        if self.staticat_config.dataset_template:
            template = custom_template(self.staticat_config.dataset_template)
        else:
            template = default_template("dataset.html")

        return template.render(dataset=self)

    def write_html(self):
        """Writes the website of the dataset to the file index.html."""
        logger.info(f"{self.log_directory}: Writing index.html")

        try:
            write(self.directory / "index.html", self.render_html())
        except Exception as error:
            raise Exception("Could not write index.html") from error

    def process(self):
        """Processes the dataset."""
        if self.should_convert_excel:
            self.convert_excel()

        self.add_distributions()
        self.write_html()


class Catalog(CatalogTOML):
    """A catalog with RDF properties, Staticat properties and processing methods."""

    def __init__(self, config):
        """Initializes the catalog, parsing and validating the file catalog.toml."""
        logger.info(f"{config.directory.name}: Parsing catalog.toml")

        try:
            with open(config.directory / "catalog.toml", "rb") as file:
                super().__init__(**tomllib.load(file))
        except Exception as error:
            raise Exception("Could not parse catalog.toml") from error

        self._staticat_config = config
        self._datasets = []

    @property
    def datasets(self):
        """The datasets belonging to the catalog (DCAT property)."""
        return self._datasets

    @property
    def directory(self):
        """The directory of the catalog."""
        return self.staticat_config.directory

    @property
    def html_description(self):
        """The description of the catalog, rendered from Markdown to HTML."""
        return MarkdownIt("js-default").render(self.description)

    @property
    def log_directory(self):
        """The directory of the catalog, formatted for logging."""
        return self.staticat_config.directory.name

    @property
    def staticat_config(self):
        """The global Staticat configuration."""
        return self._staticat_config

    @property
    def tree(self):
        """The file tree of the catalog.

        Returns an iterable of dictionaries meant to be processed in the Jinja template
        for the website of the catalog.
        """
        datasets = {dataset.relative_directory for dataset in self.datasets}
        parents = {parent for dataset in datasets for parent in dataset.parents}
        items = sorted((datasets | parents) - {Path(".")})

        for item in items:
            yield {
                "name": item.name,
                "href": quote((item / "index.html").as_posix()),
                "class": "dataset" if item in datasets else "directory",
                "depth": len(item.parents) - 1,
            }

    def add_datasets(self):
        """Adds subdirectories to the catalog as datasets.

        Only subdirectories containing a file dataset.toml are processed and added to
        the catalog.
        """
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

    def render_css(self):
        """Renders the Staticat stylesheet."""
        return default_template("default.css").render()

    def render_html(self):
        """Renders the website of the catalog."""
        if self.staticat_config.catalog_template:
            template = custom_template(self.staticat_config.catalog_template)
        else:
            template = default_template("catalog.html")

        return template.render(catalog=self)

    def render_rdf(self):
        """Renders an RDF/XML representation of the catalog."""
        return default_template("catalog.rdf").render(catalog=self)

    def write_css(self):
        """Writes the Staticat stylesheet to the file default.css."""
        logger.info(f"{self.directory.name}: Writing default.css")

        try:
            write(self.directory / "default.css", self.render_css())
        except Exception as error:
            raise Exception("Could not write default.css") from error

    def write_html(self):
        """Writes the website of the catalog to the file index.html."""
        logger.info(f"{self.directory.name}: Writing index.html")

        try:
            write(self.directory / "index.html", self.render_html())
        except Exception as error:
            raise Exception("Could not write index.html") from error

    def write_ttl(self):
        """Writes a Turtle representation of the catalog to the file catalog.ttl."""
        logger.info(f"{self.directory.name}: Writing catalog.ttl")

        try:
            graph = Graph()
            graph.parse(format="xml", data=self.render_rdf())
            graph.serialize(self.directory / "catalog.ttl", encoding="utf-8")
        except Exception as error:
            raise Exception("Could not write catalog.ttl") from error

    def process(self):
        """Processes the catalog."""
        logger.info(f"{self.directory.name}: Processing catalog...")
        self.add_datasets()

        try:
            self.write_ttl()
            self.write_css()
            self.write_html()
        except Exception as error:
            logger.critical(
                f"{self.log_directory}: Could not process catalog: {error}"
                + (f": {error.__cause__}" if error.__cause__ else "")
            )

            raise Exception("Could not process catalog") from error
