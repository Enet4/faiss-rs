#!/bin/sh
repo_url=https://github.com/Enet4/faiss.git
repo_rev=c_api_head

git clone $repo_url faiss --branch $repo_rev --depth 1

# Build
cmake . -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON

make faiss_c
mkdir -p $HOME/.faiss_c
cp c_api/libfaiss_c.so $HOME/.faiss_c/

echo "libfaiss_c.so installed in $HOME/.faiss_c/"

cd ..

# clean up
rm -rf faiss
