EXAMPLE_PATH=./var/stimuli

all:
	@echo OK

pmsp-1996:
	time bin/pmsp.py pmsp-1996 --retrain

adkp-2017:
	time bin/pmsp.py adkp-2017 --retrain

mdlpa-2020:
	time bin/pmsp.py mdlpa-2020 --retrain

vowels-for-word-learning:
	time bin/pmsp.py vowels-for-word-learning --retrain
 
lens-stimuli:
	mkdir -p $(EXAMPLE_PATH)

	bin/pmsp.py lens-stimuli \
		--wordfile pmsp/data/plaut_dataset_collapsed.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/pmsp-train.ex

	bin/pmsp.py lens-stimuli \
		--wordfile pmsp/data/anchors_new1.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n1.ex

	bin/pmsp.py lens-stimuli \
		--wordfile pmsp/data/anchors_new2.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n2.ex

	bin/pmsp.py lens-stimuli \
		--wordfile pmsp/data/anchors_new3.csv \
		--freqfile pmsp/data/word-frequencies.csv \
		--outfile $(EXAMPLE_PATH)/anchors-n3.ex

	bin/pmsp.py lens-stimuli \
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
	rm -rf build dist pmsp_torch.egg-info
	# rm -rf var/stimuli var/results

install:
	python ./setup.py install

notebook:
	jupyter notebook

.PHONY: all requirements clean lens-stimuli run clean install notebook
