#!/bin/bash -e

function generate_api () {
    # generate_api $MEGENGINE_PYTHONPATH $BUILD_LANG $API_DIR

    pushd $(dirname $0)
    rm -rf ../build_api/html

    eval MEGENGINE_PYTHONPATH="$1"
    eval BUILD_LANG="$2"
    eval API_DIR="$3"

    AUTOGEN=../source_api/${API_DIR}

    rm -rf $AUTOGEN

    export SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance"

    sphinx-apidoc -t templates -M -o $AUTOGEN $(realpath $MEGENGINE_PYTHONPATH)

    tail -n +4 $AUTOGEN/megengine.data.transform.rst >> $AUTOGEN/megengine.data.rst
    rm $AUTOGEN/megengine.data.transform.rst
    tail -n +4 $AUTOGEN/megengine.data.transform.vision.rst >> $AUTOGEN/megengine.data.rst
    rm $AUTOGEN/megengine.data.transform.vision.rst
    tail -n +4 $AUTOGEN/megengine.data.dataset.rst >> $AUTOGEN/megengine.data.rst
    rm $AUTOGEN/megengine.data.dataset.rst
    tail -n +4 $AUTOGEN/megengine.data.dataset.vision.rst >> $AUTOGEN/megengine.data.rst
    rm $AUTOGEN/megengine.data.dataset.vision.rst

    # to avoid warning for unreferenced file
    rm -f $AUTOGEN/modules.rst

    popd

    sphinx-build ${BUILD_LANG} -j$(nproc) source_api build_api/html
}

# clear cached files
rm -rf source_api/en/api
rm -rf source_api/zh/api
rm -rf source_api/include/file
rm -rf source_api/doxyoutput
rm -rf source_api/cpp_api

# get megengine python path
if [ ! -n "$2" ]; then
    MEGENGINE_PYTHONPATH=`python3 -c "import os; \
                          import megengine; \
                          print(os.path.dirname(megengine.__file__))"`
else
    MEGENGINE_PYTHONPATH=$2
fi

# generate English python docstring
BUILD_LANG_EN="-D language=en_US"
API_DIR_EN="en/api"
generate_api $MEGENGINE_PYTHONPATH "\${BUILD_LANG_EN}" $API_DIR_EN

rm -rf en_python_doc
mkdir en_python_doc
cp -r build_api/html/en en_python_doc

MEGENGINE_SOURCE_ROOT="$1"
cat source_api/include/h_file.txt | while read line; do
    line2=${line#*include}
    mkdir -p "source_api/include/file${line2%/*}"
    cp ${MEGENGINE_SOURCE_ROOT}${line} "source_api/include/file${line2%/*}"
done

# generate Chinese python docstring
export BUILD_LANG_CN="-D language=zh_CN"
API_DIR_CN="zh/api"
generate_api $MEGENGINE_PYTHONPATH "\${BUILD_LANG_CN}" $API_DIR_CN

rm -rf build_api/html/en
mkdir build_api/html/en
cp -r en_python_doc/en/* build_api/html/en
rm -rf en_python_doc

python3 gen_docs/gen_label.py api build_api/html/label.json
