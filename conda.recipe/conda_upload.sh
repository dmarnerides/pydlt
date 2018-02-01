
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export GIT_BUILD_STR=$(git describe --always)
export VERSION=$(cat .version_str)
conda build conda.recipe --no-test
anaconda -t $CONDA_UPLOAD_TOKEN upload -u demetris -l nightly $CONDA_BLD_PATH/noarch/pydlt-$VERSION-0.tar.bz2 --force