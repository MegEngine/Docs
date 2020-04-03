# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'MegEngine Documents'
copyright = '2020, Megvii'
author = 'Megvii'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinxcontrib.jupyter',
    'sphinx_autodoc_typehints',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
locale_dirs = ['locale/']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# --  Options for jupyter output ---------------------------------------------------

jupyter_kernels = {
    'python3': {
        'kernelspec': {
            'display_name': 'Python',
            'language': 'python3',
            'name': 'python3'
        },
        'file_extension': '.py'
    },
}

# -- Options for skipping specific docs --------------------------------------

skip_blacklist = frozenset(
    [
        '__weakref__', '__module__', '__doc__', '__abstractmethods__',
        '__hyperparam_spec__', '__hyperparam_trans_dict__', '__param_init_spec__'
    ]
)
skip_whitelist = frozenset(['', ''])


def handle_skip(app, what, name, obj, skip, options):
    if name.startswith('_abc_') or name in skip_blacklist:
        return True
    if name.startswith('__testcase'):
        return False
    if (name.startswith('__') and name.endswith('__') and
            getattr(obj, '__doc__', None)):
        return False
    return skip


# -- Options for doctest -----------------------------------------------------

doctest_global_setup = '''
import numpy as np
np.random.seed(0)
import megengine as mge
np.set_printoptions(precision=4)
'''


def setup(app):
    app.connect("autodoc-skip-member", handle_skip)
