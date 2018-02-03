
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export GIT_BUILD_STR=$(git describe --always)
export VERSION=$(cat .version)
conda build conda --no-test
anaconda -t $CONDA_UPLOAD_TOKEN upload -u demetris -l nightly $CONDA_BLD_PATH/noarch/pydlt-$VERSION-$GIT_BUILD_STR.tar.bz2 --force