all: test
	@echo OK

test:
	bin/pmsp.py test

run:
	bin/pmsp.py simulate

view:
	bin/pmsp.py generate

write:
	bin/pmsp.py generate --write

requirements:
	pip install -r requirements.txt

clean:
	rm -rf build dist pmsp_torch.egg-info var/stimuli var/results
	-rmdir var

install:
	python ./setup.py install

.PHONY: all requirements clean view write test
