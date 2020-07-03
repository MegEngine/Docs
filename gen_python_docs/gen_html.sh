#!/bin/bash

rm -rf source/api
rm -rf source/api_zh
rm -rf source/include/file
rm -rf source/doxyoutput
rm -rf source/cpp_api

set -e

if [ ! -n "$1" ]
then
    echo "MegBrain directory not provided"
    exit 1
else
    MGB_ROOT=$1
fi

exec 3<"source/include/h_file.txt"
exec 4<"source/include/h_location.txt"

while read line1<&3 && read line2<&4
do
    mkdir -p "source/include/file${line2%/*}"
    cp ${MGB_ROOT}${line1} "source/include/file${line2%/*}"
done

if [ ! -n "$2" ]; then
    MGE_ROOT=`python3 -c "import os; \
                          import megengine; \
                          print(os.path.dirname(megengine.__file__))"`
else
    MGE_ROOT=$2
fi

./gen_python_docs/gendoc.sh $MGE_ROOT

if [[ ! -f .tmp ]]
then
    mkdir .tmp
fi

cp -r build/html/api/* .tmp/
export BUILD_LANG="zh_CN"
export API_DIR="api_zh"
./gen_python_docs/gendoc.sh $MGE_ROOT
rm -rf build/html/api
mkdir build/html/api
cp .tmp/* build/html/api
rm -rf .tmp
