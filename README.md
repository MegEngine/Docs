# MegEngine Documents

## Prerequisites

- Install `sphinx>=2.0` and related dependencies by:
    ```
    pip3 install -U sphinx sphinx-autodoc-typehints sphinx-serve sphinxcontrib-jupyter nbsphinx jieba
    pip3 install git+https://github.com/pandas-dev/pydata-sphinx-theme.git@master
    ```
- reStructuredText (RST) is used for document writing. HTML files can be generated from the RST files for document visualization.

    For more information about RST, please visit https://sphinx-doc-zh.readthedocs.io/en/latest/rest.html.

## Generate API document

1. Make sure you have installed [MegEngine](https://github.com/MegEngine/MegEngine).

    ```bash
    pip3 install megengine -f https://megengine.org.cn/whl/mge.html
    ```

2. Run [gen_python_docs/gendoc.sh](gen_python_docs/gendoc.sh) to generate HTML files.
    The script accepts the previous python `site-packages` directory as the argument.
    Default value is `~/.local/lib/python3.6/site-packages`.
    Note that the RST files generated from python docstring are put under `source/autogen`.
    ```bash
    ./gen_python_docs/gendoc.sh ~/.local/lib/python3.6/site-packages
    ```

3. Start local sphinx service by:
    ```bash
    sphinx-serve -b build -p 8000
    ```

## Write python API document

* How documents are generated for python codes
    1. Write comments following docstring rules.
    2. Run sphinx tool to generate RST files from python docstring.
    3. Generate HTML files from RST.

    Refer to [gen_python_docs/gendoc.sh](gen_python_docs/gendoc.sh) for more details.

* Example python docstring: see [gen_python_docs/example/example.py](gen_python_docs/example/example.py).

## Run doctest in API document

API docstring also contains examples written by [doctest](https://docs.python.org/3/library/doctest.html). Run the tests by

```
gen_python_docs/gene_html.sh ~/.local/lib/python3.6/site-packages
sphinx-build -b doctest source build/doctest
```

If all tests are passed, you shall see the following similar printouts:

```
Doctest summary
===============
   16 tests
    0 failures in tests
    0 failures in setup code
    0 failures in cleanup code
build succeeded.
```

Otherwise, please fix any failed test or warning.
