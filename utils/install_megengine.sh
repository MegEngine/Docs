#!/bin/bash

CUR_DIR="$( cd "$(dirname $0)" >/dev/null 2>&1 ; pwd -P )"

pushd ${CUR_DIR}/../../ >/dev/null

    # find latest version of MegEngine
    MGE_VER=$(aws --endpoint-url=http://oss-internal.hh-b.brainpp.cn s3 ls s3://megbrain-integraion-test/wheels/megengine/ | awk '{split($0, a, "-"); print a[4]}' | sed 's/[a-z+].*//' | uniq | sort -V | tail -n 1)

    # install MegEngine, kai has configured for oss, kai has python3.8
    echo "installing newest version of megengine..."
    MEGENGINE_WHEEL=MegEngine-${MGE_VER}-cp38-cp38-manylinux2010_x86_64.whl
    aws --endpoint-url=http://oss-internal.hh-b.brainpp.cn s3 cp s3://megbrain-integraion-test/wheels/megengine/${MEGENGINE_WHEEL} .
    python3 -m pip install ${MEGENGINE_WHEEL}

    # delete wheel file
    rm -f ${MEGENGINE_WHEEL}

    # remove old MegEngine repo
    rm -rf MegEngine
    # clone MegEngine from github, assume github master branch always contains the latest release
    git clone https://github.com/MegEngine/MegEngine.git

popd >/dev/null
