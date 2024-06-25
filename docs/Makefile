# Makefile for the LCC documentation

DIR=$(shell pwd)

all:
	(make doxy)	

clean:
	rm -f *.backup

doxy:
	(doxygen ./source-doxygen/Doxyfile)	
	(cd ./build-doxygen/latex ; make ; cp refman.pdf ../../sedacs-doxygen.pdf)	

sphinx:
	(cp ../README.md ./source-sphinx/)
	(cp ../test/README.md ./source-sphinx/README-test.md)
	(cp ./README.rst ./source-sphinx/README-docs.rst)
	(cd ./source-sphinx/; make latex)
	(cd ./build-sphinx/latex/; make; cp sedacs.pdf ../../sedacs-sphinx.pdf)

