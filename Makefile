EXAMPLE_PATH=./var/stimuli

all:
	@echo OK

pmsp-1996:
	time src/scripts/pmsp-cli.py pmsp-1996 --train

adkp-2017:
	time src/scripts/pmsp-cli.py adkp-2017 --train

mdlpa-2020:
	time src/scripts/pmsp-cli.py mdlpa-2020 --train

vowels-for-word-learning:
	@echo to train, make pmsp-1996 first or invoke with --train
	time src/scripts/pmsp-cli.py vowels-for-word-learning --no-train
 
generate-the-normalized:
	src/scripts/pmsp-cli.py generate \
		--thenormalized \
		--wordfile src/pmsp/data/plaut_dataset_collapsed.csv \
		--freqfile src/pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/pmsp-train-the-normalized.ex

generate-lens-stimuli:
	mkdir -p $(EXAMPLE_PATH)

	src/scripts/pmsp-cli.py lens-stimuli \
		--wordfile src/pmsp/data/plaut_dataset_collapsed.csv \
		--freqfile src/pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/pmsp-train.ex

	src/scripts/pmsp-cli.py lens-stimuli \
		--wordfile src/pmsp/data/anchors_new1.csv \
		--freqfile src/pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n1.ex

	src/scripts/pmsp-cli.py lens-stimuli \
		--wordfile src/pmsp/data/anchors_new2.csv \
		--freqfile src/pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n2.ex

	src/scripts/pmsp-cli.py lens-stimuli \
		--wordfile src/pmsp/data/anchors_new3.csv \
		--freqfile src/pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n3.ex

	src/scripts/pmsp-cli.py lens-stimuli \
		--wordfile src/pmsp/data/probes_new.csv \
		--freqfile src/pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/probes-new.ex

	cat $(EXAMPLE_PATH)/pmsp-train.ex $(EXAMPLE_PATH)/anchors-n1.ex > \
		$(EXAMPLE_PATH)/pmsp-added-anchors-n1.ex

	cat $(EXAMPLE_PATH)/pmsp-train.ex $(EXAMPLE_PATH)/anchors-n2.ex > \
		$(EXAMPLE_PATH)/pmsp-added-anchors-n2.ex

	cat $(EXAMPLE_PATH)/pmsp-train.ex $(EXAMPLE_PATH)/anchors-n3.ex > \
		$(EXAMPLE_PATH)/pmsp-added-anchors-n3.ex

requirements:
	pip install -r src/requirements.txt

clean:
	rm -rf src/build src/dist src/pmsp_torch.egg-info
	# rm -rf var/stimuli var/results

install:
	cd src && python setup.py install

notebook:
	jupyter notebook

.PHONY: all requirements clean lens-stimuli run clean install notebook
