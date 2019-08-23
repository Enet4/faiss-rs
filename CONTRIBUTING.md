# Contributing

Faiss-rs accepts outside contributions. Please attend to the following points specific to this repository, in order to make the process as smooth as possible:

- The `faiss` crate only contains high-level bindings to Faiss, it should not be confused with the main Faiss project. If you have an issue that fundamentally applies to the main Faiss project, please file it at [their repository](https://github.com/facebookresearch/faiss) instead.
- Requests to add more features to these bindings are acceptable. Once evaluated, the maintainer may well suggest you to work on it yourself with their mentorship. Some features may also require the C API to be expanded too. The crate maintainer may reply whether it is the case, and provide some guidance on developing for the C API.
- When contributing with code, please remember to document new public types and functions, run all tests (`cargo test`) and keep code well formated with `rustfmt`. If you have an GPU with CUDA support, you may wish to test with the `gpu` flag enabled as well.
- In order to update the low-level bindings to Faiss:
    1. You will need to [install Rust bindgen](https://rust-lang.github.io/rust-bindgen/requirements.html) first.
    2. Open "gen_bindings.sh" and edit the variables `repo_url` and `repo_rev` to point to the revision of choice;
    3. Run "gen_bindings.sh";
    4. If all is OK, please don't forget to update the same variables in "travis/install_faiss_c.sh".
