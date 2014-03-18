all:	arabic_codeswitch.pdf

arabic_codeswitch.pdf: arabic_codeswitch.tex 
	pdflatex arabic_codeswitch.tex
	bibtex arabic_codeswitch
	pdflatex arabic_codeswitch.tex
	pdflatex arabic_codeswitch.tex
	pdflatex arabic_codeswitch.tex
	


clean:	
	rm -f *.lot *.lof *.toc *.aux *.bbl *.log *.dvi *.blg
	rm -f arabic_codeswitch.pdf
