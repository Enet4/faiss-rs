#!/bin/sh
repo_url=https://github.com/Enet4/faiss.git
repo_rev=2ac91ad79d9b82800804e073b13a64223cdd6727

git clone $repo_url faiss
cd faiss
git checkout -q $repo_rev

# Build
./configure --without-cuda

make libfaiss.a
cd c_api
make libfaiss_c.so
mkdir -p $HOME/.faiss_c
cp libfaiss_c.so $HOME/.faiss_c/

echo libfaiss_c.so installed in $HOME/.faiss_c/

cd ..

# clean up
rm -rf faiss
