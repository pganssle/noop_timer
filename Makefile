build: VERSION
	rm -rf dist
	tox -e build -- dist

sign: VERSION
	# Sign
	for f in dist/*; do \
		gpg --armor --output $$f.asc --detach-sig $$f ; \
	done

dist/.build_complete: VERSION
	make build
	make sign
	touch dist/.build_complete

release_pypi: VERSION dist/.build_complete
	env $$(gpg -d passfile.gpg 2>/dev/null) tox -e release -- -r pypi

release_test: VERSION dist/.build_complete
	env $$(gpg -d passfile.gpg 2>/dev/null) tox -e release

clean:
	rm -rf dist/

.PHONY: build sign release_pypi release_test clean
