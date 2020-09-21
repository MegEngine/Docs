#!/bin/bash -e

cd $(dirname $0)
rm -rf ../build_api/html

AUTOGEN=../source_api/zh/api
rm -rf $AUTOGEN

if [ ! -n "$1" ]; then
    MGE_ROOT=`python3 -c "import os; \
                          import megengine; \
                          print(os.path.dirname(megengine.__file__))"`
else
    MGE_ROOT=$1
fi

export SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance"

for i in megengine
do
    sphinx-apidoc -t templates -M -o $AUTOGEN $(realpath $MGE_ROOT)/$i
done

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

cd ../source_api

# generate pot file
sphinx-build -b gettext . ../build_api/gettext

# generate/update po file
sphinx-intl update -p ../build_api/gettext -l zh_CN -l en
