LENS_WORKPATH=~/.lens-storage/quasiregularity/frequency

all: test
	@echo OK

test:
	bin/pmsp.py test

run:
	bin/pmsp.py simulate

view:
	bin/pmsp.py generate

stimuli:
	bin/pmsp.py generate \
		--infile PMSP/data/pmsp-data.csv \
		--outfile $(LENS_WORKPATH)/pmsp-train.ex

	bin/pmsp.py generate \
		--infile PMSP/data/anchors_new1.csv \
		--outfile $(LENS_WORKPATH)/anchors-n1.ex

	bin/pmsp.py generate \
		--infile PMSP/data/anchors_new2.csv \
		--outfile $(LENS_WORKPATH)/anchors-n2.ex

	bin/pmsp.py generate \
		--infile PMSP/data/anchors_new3.csv \
		--outfile $(LENS_WORKPATH)/anchors-n3.ex

	bin/pmsp.py generate \
		--infile PMSP/data/probes_new.csv \
		--outfile $(LENS_WORKPATH)/probes-new.ex

	cat $(LENS_WORKPATH)/pmsp-train.ex $(LENS_WORKPATH)/anchors-n1.ex > \
		$(LENS_WORKPATH)/pmsp-added-anchors-n1.ex

	cat $(LENS_WORKPATH)/pmsp-train.ex $(LENS_WORKPATH)/anchors-n2.ex > \
		$(LENS_WORKPATH)/pmsp-added-anchors-n2.ex

	cat $(LENS_WORKPATH)/pmsp-train.ex $(LENS_WORKPATH)/anchors-n3.ex > \
		$(LENS_WORKPATH)/pmsp-added-anchors-n3.ex

requirements:
	pip install -r requirements.txt

clean:
	rm -rf build dist pmsp_torch.egg-info var/stimuli var/results
	-rmdir var

install:
	python ./setup.py install

.PHONY: all requirements clean view write test
