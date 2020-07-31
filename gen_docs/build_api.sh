#!/bin/bash -e

rm -rf source_api/api
rm -rf source_api/api_zh
rm -rf source_api/include/file
rm -rf source_api/doxyoutput
rm -rf source_api/cpp_api


if [ ! -n "$2" ]; then
    MGE_ROOT=`python3 -c "import os; \
                          import megengine; \
                          print(os.path.dirname(megengine.__file__))"`
else
    MGE_ROOT=$2
fi

export BUILDLANG="-D language=en_US"
./gen_docs/build.sh $MGE_ROOT

rm -rf en_python_doc
mkdir en_python_doc
cp -r build_api/html/en en_python_doc


MGB_ROOT="$1"
cat source_api/include/h_file.txt | while read line; do
    line2=${line#*include}
    mkdir -p "source_api/include/file${line2%/*}"
    cp ${MGB_ROOT}${line} "source_api/include/file${line2%/*}"
done

export API_DIR="zh/api"
# generate Chinese python document, tutorial, and English c++ document
export BUILDLANG="-D language=zh_CN"
./gen_docs/build.sh $MGE_ROOT

rm -rf build_api/html/en
mkdir build_api/html/en
cp -r en_python_doc/en/* build_api/html/en
rm -rf en_python_doc

python3 gen_docs/gen_label.py api build_api/html/label.json

