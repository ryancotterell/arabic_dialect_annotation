all:	dialect_annotation.pdf

dialect_annotation.pdf:	dialect_annotation.tex dialect_annotation.bib
	pdflatex dialect_annotation
	bibtex dialect_annotation
	pdflatex dialect_annotation
	pdflatex dialect_annotation
	pdflatex dialect_annotation


clean:	
	rm -f *.lot *.lof *.toc *.aux *.bbl *.log *.dvi *.blg
	rm -f dialect_annotation.pdf
