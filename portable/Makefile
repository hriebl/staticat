.PHONY: all clean dist/staticat

TEMPLATES = dist/templates/catalog.html dist/templates/dataset.html

OPENDATA = dist/opendata/catalog.toml \
	dist/opendata/example/distribution.xlsx \
	dist/opendata/example/dataset.toml \
	dist/opendata/online/dataset.toml \
	dist/opendata/topic/example2/dataset.toml \
	dist/opendata/topic/example2/distribution.csv \
	dist/opendata/topic/example1/dataset.toml \
	dist/opendata/topic/example1/distribution.csv

all: dist/staticat dist/config.toml $(OPENDATA) $(TEMPLATES)

clean:
	rm -rf build dist staticat.spec

dist/staticat: staticat logo.ico
	pyinstaller --collect-data staticat --icon logo.ico --noconfirm --onefile $<

dist/config.toml: config.toml
	@mkdir -p $(@D)
	cp $< $@

$(TEMPLATES): dist/%: ../src/staticat/%
	@mkdir -p $(@D)
	cp $< $@

$(OPENDATA): dist/%: ../%
	@mkdir -p $(@D)
	cp $< $@
