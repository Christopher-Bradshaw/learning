files = $(basename *.tex))

default: $(addsuffix pdf, $(files))

%.pdf: %.tex
	tex $(basename $@) && dvipdfm $(basename $@)

.PHONY: clean
clean:
	rm $(addsuffix dvi, $(files))
	rm $(addsuffix log, $(files))
	rm $(addsuffix pdf, $(files))

