#!/bin/bash -e

if [ ! -n "$2" ]; then
    ./gen_docs/build_api.sh $1
else
    ./gen_docs/build_api.sh $1 $2
fi

echo "api pages have been generated"

./gen_docs/build_doc.sh

echo "doc pages have been generated"

./gen_docs/build_search.sh

echo "global search index has been generated"

rm -rf build

mkdir -p build/html/
mkdir -p build/html/api/latest/
mkdir -p build/html/doc

cp -r build_api/html/* build/html/api/latest/
cp -r build_doc/html/* build/html/doc