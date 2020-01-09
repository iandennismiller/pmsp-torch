all: test
	@echo OK

test:
	bin/pmsp.py test

run:
	bin/pmsp.py simulate

view:
	bin/pmsp.py generate

stimuli:
	bin/pmsp.py generate --write \
		--infile pmsp-data.csv \
		--outfile pmsp-train.ex

	bin/pmsp.py generate --write \
		--infile anchors_new1.csv \
		--outfile anchors-n1.ex

	bin/pmsp.py generate --write \
		--infile anchors_new2.csv \
		--outfile anchors-n2.ex

	bin/pmsp.py generate --write \
		--infile anchors_new3.csv \
		--outfile anchors-n3.ex

	bin/pmsp.py generate --write \
		--infile probes_new.csv \
		--outfile probes-new.ex

requirements:
	pip install -r requirements.txt

clean:
	rm -rf build dist pmsp_torch.egg-info var/stimuli var/results
	-rmdir var

install:
	python ./setup.py install

.PHONY: all requirements clean view write test
