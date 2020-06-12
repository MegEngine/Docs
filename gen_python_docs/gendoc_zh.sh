#!/bin/bash -e

cd $(dirname $0)
rm -rf ../build/html

AUTOGEN=../source/api_zh
rm -rf $AUTOGEN

ROOT_PATH=$1

#if [ ! -f "$ROOT_PATH/megengine/example.py" ]; then
    #ln -s $PWD/example/example.py $ROOT_PATH/megengine/
#fi

export SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance"

for i in megengine
do
    sphinx-apidoc -t templates -M -o $AUTOGEN $(realpath $ROOT_PATH)/$i
done

tail -n +4 $AUTOGEN/megengine.data.transform.rst >> $AUTOGEN/megengine.data.rst
rm $AUTOGEN/megengine.data.transform.rst
tail -n +4 $AUTOGEN/megengine.data.transform.vision.rst >> $AUTOGEN/megengine.data.rst
rm $AUTOGEN/megengine.data.transform.vision.rst
tail -n +4 $AUTOGEN/megengine.data.dataset.rst >> $AUTOGEN/megengine.data.rst
rm $AUTOGEN/megengine.data.dataset.rst
tail -n +4 $AUTOGEN/megengine.data.dataset.vision.rst >> $AUTOGEN/megengine.data.rst
rm $AUTOGEN/megengine.data.dataset.vision.rst

# add contents on each page
# sed -e '9i.. contents::\n' $AUTOGEN/* -i

# add imported-members on each module
# sed -e '/:members:/a\ \ \ \ :imported-members:' $AUTOGEN/* -i

# fix title level
# sed -e '/ module$/ {n; s/-/^/g}' $AUTOGEN/* -i

# to avoid warning for unreferenced file
rm -f $AUTOGEN/modules.rst

cd ..
sphinx-build -D language="zh_CN" source build/html