#!/bin/sh
repo_url=https://github.com/Enet4/faiss.git
repo_rev=78bd95d45bb87d33dc47bfd5d27353635cdfcca2

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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.faiss_c/

cd ..

# clean up
rm -rf faiss
