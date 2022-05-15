#!/bin/sh
if ! which bindgen > /dev/null; then
    echo "ERROR: `bindgen` not found. Please install using cargo:"
    echo "    cargo install bindgen"
    exit 1
fi

repo_url=https://github.com/facebookresearch/faiss
repo_rev=v1.7.2
cuda_root=/opt/cuda

git clone "$repo_url" faiss --branch "$repo_rev" --depth 1

bindgen_opt='--size_t-is-usize --whitelist-function faiss_.* --whitelist-type idx_t|Faiss.* --opaque-type FILE'

headers=`ls faiss/c_api/*_c.h faiss/c_api/impl/*_c.h faiss/c_api/utils/*_c.h`
echo '// Auto-generated, do not edit!' > c_api.h
for header in $headers; do
    echo "#include \""$header"\"" >> c_api.h;
done

cmd="bindgen --rust-target 1.33 $bindgen_opt c_api.h -o src/bindings.rs"
echo ${cmd}
${cmd}

headers=faiss/c_api/gpu/*_c.h
for header in $headers; do
    echo "#include \""$header"\"" >> c_api.h;
done

cmd="bindgen --rust-target 1.33 $bindgen_opt c_api.h -o src/bindings_gpu.rs -- -Ifaiss/c_api -I$cuda_root/include"
echo ${cmd}
${cmd}

# clean up
rm -rf faiss
rm -f c_api.h
