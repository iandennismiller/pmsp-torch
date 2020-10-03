EXAMPLE_PATH=./var/stimuli

all:
	@echo OK

replicate:
	time bin/pmsp.py adkp-2017 --retrain

run:
	time bin/pmsp.py inspect-vowel-activation
 
retrain:
	time bin/pmsp.py inspect-vowel-activation --retrain

generate:
	mkdir -p $(EXAMPLE_PATH)

	bin/pmsp.py generate \
		--wordfile pmsp/data/plaut_dataset_collapsed.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/pmsp-train.ex

	bin/pmsp.py generate \
		--wordfile pmsp/data/anchors_new1.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n1.ex

	bin/pmsp.py generate \
		--wordfile pmsp/data/anchors_new2.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n2.ex

	bin/pmsp.py generate \
		--wordfile pmsp/data/anchors_new3.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n3.ex

	bin/pmsp.py generate \
		--wordfile pmsp/data/probes_new.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/probes-new.ex

	cat $(EXAMPLE_PATH)/pmsp-train.ex $(EXAMPLE_PATH)/anchors-n1.ex > \
		$(EXAMPLE_PATH)/pmsp-added-anchors-n1.ex

	cat $(EXAMPLE_PATH)/pmsp-train.ex $(EXAMPLE_PATH)/anchors-n2.ex > \
		$(EXAMPLE_PATH)/pmsp-added-anchors-n2.ex

	cat $(EXAMPLE_PATH)/pmsp-train.ex $(EXAMPLE_PATH)/anchors-n3.ex > \
		$(EXAMPLE_PATH)/pmsp-added-anchors-n3.ex

requirements:
	pip install -r requirements.txt

clean:
	rm -rf build dist pmsp_torch.egg-info var/stimuli var/results
	-rmdir var

install:
	python ./setup.py install

.PHONY: all requirements clean generate run clean install
