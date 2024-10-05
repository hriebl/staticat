import importlib
import xml.etree.ElementTree as ET
from enum import StrEnum

import pandas as pd
from rdflib import Graph
from rdflib.namespace import DC, RDF, SKOS


def read_rdf(name, file):
    path = importlib.resources.files() / file

    graph = Graph()
    graph.parse(path)
    members = {}

    for concept in graph.subjects(RDF.type, SKOS.Concept):
        identifier = str(graph.value(concept, DC.identifier))
        members[identifier] = identifier

    return StrEnum(name, members)


Availability = read_rdf("Availability", "availability.rdf")
DataTheme = read_rdf("DataTheme", "data-theme.rdf")
License = read_rdf("License", "license.rdf")


def read_file_type():
    path = importlib.resources.files() / "file-type.xml"

    tree = ET.parse(path)
    root = tree.getroot()

    data = {
        "extension": [],
        "type": [],
        "code": [],
    }

    for record in root.findall("record"):
        for extension in record.findall("file-extension"):
            data["extension"].append(extension.text)
            data["type"].append(record.findtext("internet-media-type", default=""))
            data["code"].append(record.findtext("authority-code", default=""))

    df = pd.DataFrame(data)
    duplicated = df["extension"].duplicated(keep=False)
    default = df["extension"].str.lower() == "." + df["code"].str.lower()
    df = df[~duplicated | default]
    df = df.set_index("extension")

    return df


FileTypeDF = read_file_type()


def file_type_df_to_enum():
    members = FileTypeDF["code"].drop_duplicates()
    return StrEnum("FileType", zip(members, members))


FileType = file_type_df_to_enum()
