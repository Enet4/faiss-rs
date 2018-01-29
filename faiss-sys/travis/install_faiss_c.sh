#!/bin/sh
repo_url=https://github.com/Enet4/faiss.git
repo_rev=daa965f2268676420db707c3256dfe7fc3aa0207

git clone $repo_url faiss
cd faiss
git checkout -q $repo_rev

# Build
cp ../travis/makefile.inc ./
make libfaiss.a
cd c_api
make libfaiss_c.so
mkdir -p $HOME/.faiss_c
cp libfaiss_c.so $HOME/.faiss_c/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.faiss_c

echo libfaiss_c.so installed in $HOME/.faiss_c/

cd ..

# clean up
rm -rf faiss
