all: doxygen sphinx

clean:
	rm -rf _build

doxygen:
	doxygen source/doxygen/Doxyfile
	cd _build/doxygen/latex && make && cp refman.pdf ../../sedacs-doxygen.pdf

sphinx:
	cp ../README.md source/sphinx/
	cp ../LICENSE.txt source/sphinx/
	cp README.rst source/sphinx/README-docs.rst
	cd source/sphinx/ && make html
	# cd source/sphinx/ && make latexpdf && cp ../../_build/sphinx/latex/sedacs.pdf ../../sedacs-sphinx.pdf
