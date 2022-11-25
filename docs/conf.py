from subprocess import check_output

import audeer


# Project -----------------------------------------------------------------
project = 'audmath'
author = 'Hagen Wierstorf'
version = audeer.git_repo_version()
title = '{} Documentation'.format(project)


# General -----------------------------------------------------------------
master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = None
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.katex',
    'matplotlib.sphinxext.plot_directive',
]
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
linkcheck_ignore = [
    'https://gitlab.audeering.com',
]
copybutton_prompt_text = r'>>> |\.\.\. |$ '
copybutton_prompt_is_regexp = True

# Disable auto-generation of TOC entries in the API
# https://github.com/sphinx-doc/sphinx/issues/6316
toc_object_entries = False

# Matplot directive settings
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ['png']
plot_rcparams = {'font.size': 13}


# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'footer_links': False,
    'logo_only': False,
}
html_context = {
    'display_github': True,
}
html_title = title
