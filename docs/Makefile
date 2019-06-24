# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python3 -c "import sys,sphinx,sphinx.cmd.build;sys.exit(sphinx.cmd.build.build_main([sys.argv[3], sys.argv[4]]))"
SPHINXPROJ    = eleanor
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	mkdir -p $(BUILDDIR)/html/_images
	mkdir -p $(BUILDDIR)/html/getting_started
	cp ../figures/eleanor.gif $(BUILDDIR)/html/_images/eleanor.gif
	cp ../docs/getting_started/customApExample.gif $(BUILDDIR)/html/getting_started/customApExample.gif
