#!/bin/bash -e

cd $(dirname $0)
rm -rf ../build_api/html

ROOT_PATH=$1
API_DIR=${API_DIR:-en/api}

AUTOGEN=../source_api/${API_DIR}

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

sphinx-build ${BUILDLANG} -j$(nproc) source_api build_api/html  