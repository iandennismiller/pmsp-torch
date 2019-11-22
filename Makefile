all: test
	@echo OK

run:
	./pmsp.py simulate

view:
	./pmsp.py generate

write:
	./pmsp.py generate --write

requirements:
	pip install -r requirements.txt

clean:
	rm -rf ./*_test??

test:
	./pmsp.py test

.PHONY: all requirements clean view write test
