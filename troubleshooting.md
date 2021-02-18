# TROUBLESHOOTING

## Build error "undefined reference to `faiss_ParameterSpace_free`"

You should update `https://github.com/facebookresearch/faiss` lib till minimal version `1.6.4`
because function `faiss_ParameterSpace_free` exists after this version.
