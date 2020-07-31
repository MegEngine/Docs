#!/bin/bash -e
rm -rf build_doc
sphinx-build ${BUILDLANG} -j$(nproc) source build_doc/html
python3 gen_docs/gen_label.py doc build_doc/html/label.json
