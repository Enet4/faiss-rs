#!/bin/sh
repo_url=https://github.com/Enet4/faiss.git
repo_rev=2bfbead8f1f29030c11797d161b0b9dec6c2d8a3

git clone $repo_url faiss
cd faiss
git checkout -q $repo_rev

# Build
./configure

make libfaiss.a
cd c_api
make libfaiss_c.so
mkdir -p $HOME/.faiss_c
cp libfaiss_c.so $HOME/.faiss_c/

echo libfaiss_c.so installed in $HOME/.faiss_c/

cd ..

# clean up
rm -rf faiss
