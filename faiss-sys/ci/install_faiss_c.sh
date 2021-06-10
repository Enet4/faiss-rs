#!/usr/bin/env bash
set -eu

repo_url=https://github.com/Enet4/faiss.git
repo_rev=c_api_head

git clone $repo_url faiss --branch $repo_rev --depth 1

mkdir -p "$HOME/.faiss_c"

cd faiss

git rev-parse HEAD > ../rev_hash

if [[ -s "$HOME/.faiss_c/rev_hash" && `diff -w -q ../rev_hash $HOME/.faiss_c/rev_hash` -eq "0" ]]; then
    echo "libfaiss_c.so is already built for revision" `cat ../rev_hash`

    # clean up
    cd ..
    rm -rf faiss rev_hash
    exit 0
fi


# Build
cmake . -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF

make
cp -f "../rev_hash" faiss/libfaiss.so c_api/libfaiss_c.so "$HOME/.faiss_c/"

echo "libfaiss_c.so (" `cat ../rev_hash` ") installed in $HOME/.faiss_c/"

cd ..

# clean up
rm -rf faiss rev_hash
