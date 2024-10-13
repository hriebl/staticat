# Staticat

<img src="https://raw.githubusercontent.com/hriebl/staticat/refs/heads/main/assets/logo.png" alt="logo" align="right" width="150">

Mit Staticat können Sie *statische Open-Data-Kataloge nach dem DCAT-AP.de-Standard* direkt auf Ihrem Rechner generieren. Dazu müssen Sie nur die Dateien, die Sie veröffentlichen möchten, in einem Ordner sammeln und mit einigen wenigen Metadaten im [TOML-Format](https://toml.io) anreichern. Nachdem Sie Staticat ausgeführt haben, können Sie den Ordner anschließend auf einen einfachen Webserver hochladen. MySQL, PHP usw. werden dabei nicht benötigt.

**English summary:** With Staticat, you can generate *static open data catalogs according to the DCAT-AP.de standard* right on your computer. All you need to do is collect the files you want to publish in a folder and enrich them with some metadata using the [TOML format](https://toml.io). After executing Staticat, you can then upload the folder to a simple web server. MySQL, PHP etc. are not required.

## Installation

Staticat kann als portables Programm verwendet werden. Laden Sie dazu einfach die [neueste Version](https://github.com/hriebl/staticat/releases/latest) für Windows herunter und entpacken Sie das ZIP-Archiv.

Alternativ, falls Sie bereits Python auf Ihrem Rechner installiert haben, können Sie Staticat auch mit dem Kommando `pip install staticat` installieren. Die Installation mit pip funktioniert auch auf macOS und Linux.

## Benutzung

Um einen eigenen Open-Data-Katalog aufzubauen, starten Sie am einfachsten mit dem Open-Data-Ordner, der mit Staticat ausgeliefert wird. Sie finden den Ordner auch [auf GitHub](https://github.com/hriebl/staticat/tree/main/opendata). Der Aufbau folgt dem Schema: ein Verzeichnis pro Datensatz, eine oder mehrere Distributionen, also Dateien, pro Datensatz. Datensätze können dabei in beliebiger Verschachtelung und Tiefe angelegt werden.

Führen Sie nun folgende Schritte aus:

1. Passen Sie zunächst die Metadaten für den gesamten Katalog in der Datei [catalog.toml](#catalogtoml) an.
2. Ersetzen und ergänzen Sie anschließend die Beispieldatensätze nach und nach mit "echten" Datensätzen. Dazu bearbeiten Sie erst die Metadaten für den jeweiligen Datensatz in der Datei [dataset.toml](#datasettoml) und legen dann die Dateien, die Sie dem Datensatz zuordnen möchten, im selben Verzeichnis ab.
3. Nachdem Sie Staticat ausgeführt haben, laden Sie Ihren Open-Data-Ordner zum Schluss auf einen Webserver hoch.

Im Folgenden wird die Funktionsweise von Staticat etwas ausführlicher beschrieben. Betrachten Sie dazu zum Beispiel diesen Open-Data-Ordner:

```
$ tree opendata
opendata                   ← Open-Data-Ordner
├── catalog.toml           ← Metadatenbeschreibung des Katalogs
└── example                ← Datensatz
    ├── dataset.toml       ← Metadatenbeschreibung des Datensatzes
    └── distribution.xlsx  ← Distribution

2 directories, 3 files
```

Nach dem DCAT-AP.de-Standard fasst ein Katalog mehrere Datensätze und ein Datensatz mehrere Distributionen, also in der Regel Dateien, zusammen. Dabei müssen sowohl der Katalog als auch die Datensätze mit einigen grundlegenden Metadaten ausgezeichnet werden. Dies geschieht über die Dateien catalog.toml ([Beispieldatei](https://github.com/hriebl/staticat/tree/main/opendata/catalog.toml), [Details](#catalogtoml)) und dataset.toml ([Beispieldatei](https://github.com/hriebl/staticat/tree/main/opendata/example/dataset.toml), [Details](#datasettoml)).

Wenn Sie nun Staticat ausführen, legt das Programm einige zusätzliche Dateien an:

```
$ tree opendata
opendata
├── catalog.toml
├── catalog.ttl            ← Katalog im Turtle-Format
├── default.css            ← Stylesheet für die Websites
├── index.html             ← Website des Katalogs
└── example
    ├── dataset.toml
    ├── distribution.csv   ← Distribution im CSV-Format
    ├── distribution.xlsx
    └── index.html         ← Website des Datensatzes

2 directories, 8 files
```

Excel-Dateien werden von Staticat standardmäßig ins CSV-Format umgewandelt. Das CSV-Format ist im Gegensatz zu den Formaten XLS und XLSX offen und nicht-proprietär und daher besser für Open-Data-Veröffentlichungen geeignet. Über die [Konfigurationsdateien](#konfigurationsdateien) lässt sich dieses Verhalten pro Datensatz anpassen.

Die Datei catalog.ttl beschreibt den Katalog im maschinenlesbaren Turtle-Format und kann durch andere Open-Data-Portale "geharvestet", also eingelesen, werden. Wenn Sie Ihren Katalog harvesten lassen, können Ihre Dateien zum Beispiel auch über das Open-Data-Portal Ihres Landes oder über [GovData](https://www.govdata.de) gefunden werden.

Die Websites des Katalogs und der Datensätze bieten eine einfache, für Menschen ansprechende Ansicht der veröffentlichten Dateien. Vorrangig ist der von Staticat erstellte Katalog aber für das Harvesting durch übergeordnete Open-Data-Portale gedacht.

## Screenshots

Eine Online-Demo finden Sie unter [hriebl.github.io/staticat](https://hriebl.github.io/staticat). Hier außerdem zwei Screenshots von Staticat:

| Katalog / Datensatz                                                                                                      | Katalog / Datensatz                                                                                                      |
| :----------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| ![Screenshot Katalog](https://raw.githubusercontent.com/hriebl/staticat/refs/heads/main/assets/screenshot-catalog.png)   | ![Screenshot Datensatz](https://raw.githubusercontent.com/hriebl/staticat/refs/heads/main/assets/screenshot-dataset.png) |

## Konfigurationsdateien

### config.toml

Die Datei config.toml muss im selben Verzeichnis wie das *portable* Programm staticat.exe liegen. Falls Sie Staticat mit pip installiert haben, also nicht die portable Version verwenden, nutzen Sie bitte die entsprechenden [Kommandozeilenargumente](#kommandozeilenargumente) anstatt der Datei config.toml.

Beispieldatei: [portable/config.toml](https://github.com/hriebl/staticat/tree/main/portable/config.toml).

Folgende Schlüssel sind zulässig:

| Schlüssel        | Typ                     | Beschreibung                                                    |
| :--------------- | :---------------------- | :-------------------------------------------------------------- |
| directory        | String                  | Basisverzeichnis des lokalen Open-Data-Ordners                  |
| catalog_template | String <br> (optional)  | Pfad zum Jinja-Template für die HTML-Darstellung des Katalogs   |
| dataset_template | String <br> (optional)  | Pfad zum Jinja-Template für die HTML-Darstellung der Datensätze |
| convert_excel    | Boolean <br> (optional) | Ob Excel zu CSV umgewandelt wird (für alle Datensätze)          |

### catalog.toml

Die Datei catalog.toml muss im Basisverzeichnis des Open-Data-Ordners liegen.

Beispieldatei: [opendata/catalog.toml](https://github.com/hriebl/staticat/tree/main/opendata/catalog.toml).

Folgende Schlüssel sind zulässig:

| Schlüssel            | Typ                   | RDF / Beschreibung                           |
| :------------------- | :-------------------- | :------------------------------------------- |
| uri                  | String                | Basis-URL des Open-Data-Katalogs im Internet |
| title                | String                | dct:title                                    |
| description          | String                | dct:description                              |
| \[publisher\]        | Table                 | dct:publisher                                |
| name                 | String                | foaf:name                                    |
| uri                  | String                | URI des Publishers                           |
| \[dataset_defaults\] | Table <br> (optional) | Standardmetadaten <br> (für alle Datensätze) |
| ...                  | ...                   | Siehe [dataset.toml](#datasettoml)           |

### dataset.toml

Jedes Verzeichnis im Open-Data-Ordner mit einer Datei dataset.toml wird von Staticat als Datensatz verarbeitet und zum Katalog hinzugefügt. Sollten Unklarheiten hinsichtlich der Bedeutung eines Schlüssel bestehen, können Sie das entsprechende RDF-Prädikat nachschlagen oder die HTML-Darstellung des Datensatzes zu Rate ziehen.

Beispieldatei: [opendata/example/dataset.toml](https://github.com/hriebl/staticat/tree/main/opendata/example/dataset.toml).

Folgende Schlüssel sind zulässig:

| Schlüssel           | Typ                     | Werte / Format                                        | RDF / Beschreibung                                                  |
| :------------------ | :---------------------- | :---------------------------------------------------- | :------------------------------------------------------------------ |
| title               | String                  |                                                       | dct:title                                                           |
| description         | String                  |                                                       | dct:description                                                     |
| keywords            | Array mit Strings       |                                                       | dcat:keyword                                                        |
| themes              | Array mit Strings       | [1] (nur Code)                                        | dcat:theme                                                          |
| issued              | Datum                   | JJJJ-MM-TT                                            | dct:issued                                                          |
| start_date          | Datum                   | JJJJ-MM-TT                                            | dct:temporal//dcat:startDate                                        |
| end_date            | Datum                   | JJJJ-MM-TT                                            | dct:temporal//dcat:endDate                                          |
| license             | String                  | [2] (nur Code)                                        | dct:license                                                         |
| availability        | String                  | [3] (nur Code)                                        | dcatap:availability                                                 |
| spatial             | String                  | [4], [5], [6], [7] <br> (komplette URI)               | dct:spatial                                                         |
| political_geocoding | String                  | [8], [9], [10], [11], [12], [13] <br> (komplette URI) | dcatde:politicalGeocodingURI <br> dcatde:politicalGeocodingLevelURI |
| \[maintainer\]      | Table                   |                                                       | dcatde:maintainer <br> dcat:contactPoint                            |
| name                | String                  |                                                       | foaf:name <br> vcard:fn                                             |
| email               | String                  |                                                       | foaf:mbox <br> vcard:hasEmail                                       |
| \[creator\]         | Table                   |                                                       | dct:creator                                                         |
| name                | String                  |                                                       | foaf:name                                                           |
| email               | String                  |                                                       | foaf:mbox                                                           |
| \[publisher\]       | Table                   |                                                       | dct:publisher                                                       |
| name                | String                  |                                                       | foaf:name                                                           |
| uri                 | String                  |                                                       | URI des Publishers                                                  |
| \[config\]          | Table <br> (optional)   |                                                       | Staticat-Einstellungen <br> (für diesen Datensatz)                  |
| convert_excel       | Boolean <br> (optional) |                                                       | Ob Excel zu CSV umgewandelt wird                                    |

[1]: https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/data-theme
[2]: https://www.dcat-ap.de/def/licenses/
[3]: https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/planned-availability
[4]: https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/continent
[5]: https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/country
[6]: https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/place
[7]: https://sws.geonames.org/
[8]: https://www.dcat-ap.de/def/politicalGeocoding/districtKey/
[9]: https://www.dcat-ap.de/def/politicalGeocoding/governmentDistrictKey/
[10]: https://www.dcat-ap.de/def/politicalGeocoding/municipalAssociationKey/
[11]: https://www.dcat-ap.de/def/politicalGeocoding/municipalityKey/
[12]: https://www.dcat-ap.de/def/politicalGeocoding/regionalKey/
[13]: https://www.dcat-ap.de/def/politicalGeocoding/stateKey/

Falls Sie eine Online-Ressource als Distribution zu einem Datensatz hinzufügen möchten, die nicht als Datei auf Ihrem Rechner vorliegt, können Sie diese ebenfalls in der Datei dataset.toml hinterlegen. Dazu müssen Sie lediglich ein Array `[[distributions]]` mit mindestens den beiden Schlüsseln `uri` und `title` zur Datei dataset.toml hinzufügen.

Beispieldatei: [opendata/online/dataset.toml](https://github.com/hriebl/staticat/tree/main/opendata/online/dataset.toml).

Folgende Schlüssel sind zulässig:

| Schlüssel             | Typ                    | Werte / Format  | RDF / Beschreibung                       |
| :-------------------- | :--------------------- | :-------------- | :--------------------------------------- |
| \[\[distributions\]\] | Array                  |                 | dcat:distribution                        |
| uri                   | String                 |                 | URI der Distribution <br> dcat:accessURL |
| title                 | String                 |                 | dct:title                                |
| modified              | Datum <br> (optional)  | JJJJ-MM-TT      | dct:modified                             |
| format                | String <br> (optional) | [14] (nur Code) | dct:format                               |
| media_type            | String <br> (optional) | MIME-Type       | dcat:mediaType                           |
| byte_size             | Float <br> (optional)  |                 | dcat:byteSize                            |

[14]: https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/file-type

## Kommandozeilenargumente

Falls Sie Staticat mit pip installiert haben, also nicht die portable Version verwenden, können Sie folgende Kommandozeilenargumente anstelle der Datei [config.toml](#configtoml) verwenden. Die portable Version unterstützt diese Argumente *nicht*:

```
$ staticat -h
usage: staticat [-h] [-c CATALOG_TEMPLATE] [-d DATASET_TEMPLATE] [-e] directory

positional arguments:
  directory             base directory of the local open data folder

options:
  -h, --help            show this help message and exit
  -c CATALOG_TEMPLATE, --catalog-template CATALOG_TEMPLATE
                        file path to a custom Jinja template for the HTML view of the catalog
  -d DATASET_TEMPLATE, --dataset-template DATASET_TEMPLATE
                        file path to a custom Jinja template for the HTML view of the datasets
  -e, --excel           do not convert Excel files to CSV. can be overridden in dataset.toml
```
