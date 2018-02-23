#!/bin/sh
if ! which bindgen > /dev/null; then
    echo "ERROR: `bindgen` not found. Please install using cargo:"
    echo "    cargo install bindgen"
    exit 1
fi

repo_url=https://github.com/Enet4/faiss.git
repo_rev=284885915c2d25cf213d6abdaeba643f80873d58
cuda_root=/opt/cuda

git clone $repo_url faiss
cd faiss
git checkout -q $repo_rev
cd ..

bindgen_opt='--whitelist-function faiss_.* --whitelist-type idx_t|Faiss.*'

headers=`ls faiss/c_api/*_c.h`
echo '// Auto-generated, do not edit!' > c_api.h
for header in $headers; do
    echo "#include \""$header"\"" >> c_api.h;
done

cmd="bindgen --link faiss_c $bindgen_opt c_api.h -o src/bindings.rs"
echo ${cmd}
${cmd}

headers=`ls faiss/c_api/gpu/*_c.h`
for header in $headers; do
    echo "#include \""$header"\"" >> c_api.h;
done

cmd="bindgen --link gpufaiss_c $bindgen_opt c_api.h -o src/bindings_gpu.rs -- -Ifaiss/c_api -I$cuda_root/include"
echo ${cmd}
${cmd}

# clean up
rm -rf faiss
rm -f c_api.h