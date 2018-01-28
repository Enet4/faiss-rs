#!/bin/sh
if ! which bindgen > /dev/null; then
    echo "ERROR: `bindgen` not found. Please install using cargo:"
    echo "    cargo install bindgen"
    exit 1
fi

repo_url=https://github.com/Enet4/faiss.git
repo_rev=78bd95d45bb87d33dc47bfd5d27353635cdfcca2

git clone $repo_url faiss
cd faiss
git checkout -q $repo_rev
cd ..

headers=`ls faiss/c_api/*_c.h`
echo '// Auto-generated, do not edit!' > c_api.h
for header in $headers; do
    echo "#include \""$header"\"" >> c_api.h;
done

cmd="bindgen --link faiss_c c_api.h -o src/bindings.rs"
echo ${cmd}
${cmd}

# clean up
rm -rf faiss
rm -f c_api.h