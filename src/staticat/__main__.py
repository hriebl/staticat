import argparse
import logging
from pathlib import Path

from . import ConfigTOML, staticat


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser("staticat")

    parser.add_argument(
        "-c",
        "--catalog-template",
        help="file path to a custom Jinja template for the HTML view of the catalog",
        type=Path,
    )

    parser.add_argument(
        "-d",
        "--dataset-template",
        help="file path to a custom Jinja template for the HTML view of the datasets",
        type=Path,
    )

    parser.add_argument(
        "-e",
        "--excel",
        help="do not convert Excel files to CSV. can be overridden in dataset.toml",
        action="store_true",
    )

    parser.add_argument(
        "directory",
        help="base directory of the local open data folder",
        type=Path,
    )

    args = parser.parse_args()

    staticat(
        ConfigTOML(
            directory=args.directory,
            catalog_template=args.catalog_template,
            dataset_template=args.dataset_template,
            convert_excel=not args.excel,
        )
    )


if __name__ == "__main__":
    main()
