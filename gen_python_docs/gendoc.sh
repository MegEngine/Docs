#!/bin/bash -e

cd $(dirname $0)
rm -rf ../build/html

ROOT_PATH=$1
API_DIR=${API_DIR:-api}

AUTOGEN=../source/${API_DIR}
rm -rf $AUTOGEN

export SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance"

sphinx-apidoc -t templates -M -o $AUTOGEN $(realpath $ROOT_PATH)

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

cd ..

if [[ ! -n $BUILD_LANG ]]; then
    sphinx-build -j$(nproc) source build/html
else
    sphinx-build -D language="zh_CN" -j$(nproc) source build/html
fi
