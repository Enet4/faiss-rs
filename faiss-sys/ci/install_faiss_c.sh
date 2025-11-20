#!/usr/bin/env bash
set -eu

repo_url=https://github.com/facebookresearch/faiss
repo_rev=v1.12.0

case "${1:-""}" in
	"avx512")
		echo "avx512 specified."
		FAISS_OPT_LEVEL="avx512";
		SIMD_SUFFIX="_avx512";
		;;
	"avx2")
		echo "avx2 specified."
		FAISS_OPT_LEVEL="avx2";
		SIMD_SUFFIX="_avx2";
		;;
	*)
		echo "building with FAISS_OPT_LEVEL=generic";
		FAISS_OPT_LEVEL="generic";
		SIMD_SUFFIX="";
		;;
esac

git clone "$repo_url" faiss --branch "$repo_rev" --depth 1

mkdir -p "$HOME/.faiss_c"

cd faiss

#git rev-parse HEAD > ../rev_hash

#if [[ -s "$HOME/.faiss_c/rev_hash" && `diff -w -q ../rev_hash $HOME/.faiss_c/rev_hash` -eq "0" ]]; then
#    echo "libfaiss_c.so is already built for revision" `cat ../rev_hash`
#
#    # clean up
#    cd ..
#    rm -rf faiss rev_hash
#    exit 0
#fi


# Build
cmake . \
    -DFAISS_ENABLE_C_API=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_OPT_LEVEL=${FAISS_OPT_LEVEL}

make

#cp -f "../rev_hash" "$HOME/.faiss_c/"
cp -f "faiss/libfaiss${SIMD_SUFFIX}.so" "$HOME/.faiss_c/"
# renamed to libfaiss_c.so but it links against libfaiss${SIMD_SUFFIX}.so
cp -f "c_api/libfaiss_c${SIMD_SUFFIX}.so" "$HOME/.faiss_c/libfaiss_c.so"

# shellcheck disable=SC2046
echo "libfaiss_c.so (" $(cat ../rev_hash) ") installed in $HOME/.faiss_c/"

cd ..

# clean up
#rm -rf faiss rev_hash
rm -rf faiss
