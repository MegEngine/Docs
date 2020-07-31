#!/bin/bash -e

search_index=$(cat build_doc/html/searchindex.js)
echo ${search_index/"Search.setIndex("/"Search.setIndex(\"doc\","} >build_doc/html/searchindex_global.js


search_index=$(cat build_api/html/searchindex.js)
echo ${search_index/"Search.setIndex("/"Search.setIndex(\"api\","} >build_doc/html/searchindex_api_global.js

cp build_doc/html/searchindex_global.js build_api/html/
cp build_doc/html/searchindex_api_global.js build_api/html/
