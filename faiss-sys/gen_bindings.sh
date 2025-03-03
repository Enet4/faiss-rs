#!/usr/bin/env sh
# Generate Rust bindings to the Faiss C API
#
# Ensure that the submodule is updated and checked out in the intended revision
if ! which bindgen > /dev/null; then
    echo "ERROR: `bindgen` not found. Please install using cargo:"
    echo "    cargo install bindgen-cli --version=^0.69"
    exit 1
fi

repo_url=https://github.com/facebookresearch/faiss
repo_rev=v1.10.0
cuda_root=/usr/local/cuda

if [ ! -d faiss ]; then
    git clone "$repo_url" faiss --branch "$repo_rev" --depth 1
fi

bindgen_opt='--allowlist-function faiss_.* --allowlist-type idx_t|Faiss.* --opaque-type FILE'

headers=`ls faiss/c_api/*_c.h faiss/c_api/impl/*_c.h faiss/c_api/utils/*_c.h`
echo '// Auto-generated, do not edit!' > c_api.h
for header in $headers; do
    echo "#include \""$header"\"" >> c_api.h;
done

cmd="bindgen --rust-target 1.84 $bindgen_opt c_api.h -o src/bindings.rs"
echo ${cmd}
${cmd}

headers=faiss/c_api/gpu/*_c.h
for header in $headers; do
    echo "#include \""$header"\"" >> c_api.h;
done

cmd="bindgen --rust-target 1.84 $bindgen_opt c_api.h -o src/bindings_gpu.rs -- -Ifaiss/c_api -I$cuda_root/include"
echo ${cmd}
${cmd}

# clean up
rm -f c_api.h
