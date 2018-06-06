#!/bin/sh
repo_url=https://github.com/Enet4/faiss.git
repo_rev=2fc1c5242804dac81cc0e850772b56d8ed91f63e

git clone $repo_url faiss
cd faiss
git checkout -q $repo_rev

# Build
./configure
echo '----- makefile.inc -----'
cat makefile.inc
echo '--- end makefile.inc ---'

make libfaiss.a
cd c_api
make libfaiss_c.so
mkdir -p $HOME/.faiss_c
cp libfaiss_c.so $HOME/.faiss_c/

echo libfaiss_c.so installed in $HOME/.faiss_c/

cd ..

# clean up
rm -rf faiss
