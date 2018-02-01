
OS=$TRAVIS_OS_NAME-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export GIT_BUILD_STR=$(git describe --always)
echo "Trying to set version"
echo $(pwd)
export VERSION=$(cat .version_str)
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u demetris -l nightly $CONDA_BLD_PATH/$OS/pydlt-$VERSION-0.tar.bz2 --force