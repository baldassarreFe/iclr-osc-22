# Docs

## Build docs

From the `docs` dir run:
```bash
make clean html
```

Important extensions:
- `sphinx.ext.autodoc`
- `sphinx.ext.autosummary`
- `sphinx.ext.napoleon`
- `sphinx_autodoc_typehints`

Important files:
- `autosummary.rst`
- `_templates/*.rst`

Good example of autosummary:
- https://sphinx-autosummary-recursion.readthedocs.io/
- https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion

Reference and examples:
- https://sphinx-themes.org/sample-sites/sphinx-rtd-theme/

Notes:
- if autosummary has issues importing non-existing stuff, remove `source/_autosummary/`
  and run `make clean html` again
- there is some issue with classes that inherit from `torch.nn.Module`

## Serve docs

From the `docs` dir run:
```bash
python -m http.server --directory build/html
```
