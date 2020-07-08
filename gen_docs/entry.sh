#!/bin/bash

rm -rf source/api
rm -rf source/api_zh
rm -rf source/include/file
rm -rf source/doxyoutput
rm -rf source/cpp_api

set -e

if [ ! -n "$2" ]; then
    MGE_ROOT=`python3 -c "import os; \
                          import megengine; \
                          print(os.path.dirname(megengine.__file__))"`
else
    MGE_ROOT=$2
fi

# generate English python document
export BUILDLANG="-D language=en_US"
./gen_docs/build.sh $MGE_ROOT

rm -rf en_python_doc
mkdir en_python_doc
cp -r build/html/api/* en_python_doc

# copy MegBrain source file to source/include
MGB_ROOT="$1"
cat source/include/h_file.txt | while read line; do
    line2=${line#*include}
    mkdir -p "source/include/file${line2%/*}"
    cp ${MGB_ROOT}${line} "source/include/file${line2%/*}"
done

export API_DIR="api_zh"

# generate Chinese python document, tutorial, and English c++ document
export BUILDLANG="-D language=zh_CN"
./gen_docs/build.sh $MGE_ROOT

# copy and replace English python document with previous one 
rm -rf build/html/api
mkdir build/html/api
cp en_python_doc/* build/html/api
rm -rf en_python_doc
