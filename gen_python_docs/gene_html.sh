#!/bin/bash


if [ ! -n "$1" ]
then
    ./gen_python_docs/gendoc.sh ~/.local/lib/python3.6/site-packages
else
    ./gen_python_docs/gendoc.sh $1
fi

if [[ ! -f .tmp ]]
then
    mkdir .tmp
fi

cp -r build/html/api/* .tmp/
if [ ! -n "$1" ]
then
    ./gen_python_docs/gendoc_zh.sh ~/.local/lib/python3.6/site-packages
else
    ./gen_python_docs/gendoc_zh.sh $1
fi
rm -rf build/html/api
mkdir build/html/api
cp .tmp/* build/html/api
rm -rf .tmp