tex_files = simplest story boxes_and_glue
latex_files = simplest_latex latex
fail_files = sorry

test = simplest

default: $(addsuffix .pdf, $(tex_files)) $(addsuffix .pdf, $(latex_files))

$(addsuffix .pdf, $(tex_files)): $(addsuffix .tex, $(tex_files))
	tex $(basename $@) && dvipdfm $(basename $@)

$(addsuffix .pdf, $(latex_files)): $(addsuffix .tex, $(latex_files))
	pdflatex $(basename $@)

.PHONY: clean
clean:
	files = tex_files + fail_files + latex_files
	rm $(addsuffix dvi, $(files))
	rm $(addsuffix log, $(files))
	rm $(addsuffix pdf, $(files))
