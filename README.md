# MegEngine Documents

## Prerequisites

- Install `sphinx>=2.0` and related dependencies by:
    ```
    pip3 install -U sphinx sphinx-autodoc-typehints sphinx-serve sphinxcontrib-jupyter nbsphinx jieba
    pip3 install git+https://github.com/pandas-dev/pydata-sphinx-theme.git@master
    ```
- reStructuredText (RST) is used for document writing. HTML files can be generated from the RST files for document visualization.

    For more information about RST, please visit https://sphinx-doc-zh.readthedocs.io/en/latest/rest.html.

- Install doxygen and exhale for C++ doc building

    Make sure you have installed necessary build tools (i.e. g++, python, cmake, flex, bison)

    Install doxygen: 
    ```
    git clone https://github.com/doxygen/doxygen.git
    cd doxygen
    mkdir build
    cd build
    cmake -G "Unix Makefiles" ..
    make
    make install
    ```
    Install exhale:
    ```
    pip install exhale
    ```

## Generate API document

1. Make sure you have installed [MegEngine](https://github.com/MegEngine/MegEngine).

    ```bash
    pip3 install megengine -f https://megengine.org.cn/whl/mge.html
    ```

2. Make sure you have cloned [MegBrain](https://git-core.megvii-inc.com/brain-sdk/MegBrain)

    ```bash
    git clone git@git-core.megvii-inc.com:brain-sdk/MegBrain.git
    ```

3. Run [gen_docs/entry.sh](gen_docs/entry.sh) to generate HTML files.
    The script accepts the MegEngine installation and MegBrain clone path as the argument.

    ```bash
    ./gen_docs/entry.sh $MGB_ROOT $MGE_ROOT(optional)
    ```

    Note that the RST files generated from python docstring are put under `source/autogen`.

4. Start local sphinx service by:
    ```bash
    sphinx-serve -b build -p 8000
    ```

## Write python API document

* How documents are generated for python codes
    1. Write comments following docstring rules.
    2. Run sphinx tool to generate RST files from python docstring.
    3. Generate HTML files from RST.

    Refer to [gen_docs/build.sh](gen_docs/build.sh) for more details.

* Example python docstring: see [gen_docs/example/example.py](gen_docs/example/example.py).

## Run doctest in API document

API docstring also contains examples written by [doctest](https://docs.python.org/3/library/doctest.html). Run the tests by

```bash
gen_docs/entry.sh $MGB_ROOT $MGE_ROOT(optional)
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

## Insert C++ doc hyperlink

1. For class referencing:

find the class rst file and copy its name and replace the doc with
```
:ref:`exhale_class_<filename without .rst>`
```

2. For file referencing:

find the file and copy its name and replace the doc with
```
:ref:`file_file_<filename>`
```

## Process of generate document
!["entry.sh process"](source/entry.png)

## Preview link

Run CI to generate preview link. Manually trigger is required.

https://oss.iap.hh-b.brainpp.cn/megengine-doc/doc/