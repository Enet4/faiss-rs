# Faiss-rs

[![faiss at crates.io](https://img.shields.io/crates/v/faiss.svg)](https://crates.io/crates/faiss)
[![Continuous integration status](https://github.com/Enet4/faiss-rs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Enet4/faiss-rs/actions/workflows/ci.yml)
![Minimum Rust Version Stable](https://img.shields.io/badge/Minimum%20Rust%20Version-stable-green.svg)
[![dependency status](https://deps.rs/repo/github/Enet4/faiss-rs/status.svg)](https://deps.rs/repo/github/Enet4/faiss-rs)

This project provides Rust bindings to [Faiss](https://github.com/facebookresearch/faiss),
the state-of-the-art vector search and clustering library.

## Installing with dynamic linking

By default, this crate is dynamically linked with the Faiss library installed in your system,
so it does not build Faiss automatically for you.
To build the library yourself:

  1. Follow the [instructions to build Faiss using CMake](https://github.com/Enet4/faiss/tree/c_api_head/INSTALL.md#step-1-invoking-cmake),
     enabling the variables `FAISS_ENABLE_C_API` and `BUILD_SHARED_LIBS`.
     The crate is currently only compatible with version v1.7.2.
     Consider building Faiss from [this fork, `c_api_head` branch](https://github.com/Enet4/faiss/tree/c_api_head),
     which will contain the latest supported bindings to the C interface.
     For example:

     ```sh
     cmake -B build -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
     cmake --build build
     ```

     This will result in the dynamic library `faiss_c` ("c_api/libfaiss_c.so" on Linux),
     which needs to be installed in a place where your system will pick up
     (in Linux, try somewhere in the `LD_LIBRARY_PATH` environment variable, such as "/usr/lib",
     or try adding a new path to this variable. For mac users, this means placing it in `/usr/local/lib/libfaiss_c.dylib`).
     For GPU support, don't forget to enable the option `FAISS_ENABLE_GPU`.
     **Note:** `faiss_c` might link dynamically to the native `faiss` library,
     which in that case you will need to install the main shared object (faiss/libfaiss.so)
     as well.
  2. You are now ready to include this crate as a dependency:

     ```toml
     [dependencies]
     "faiss" = "0.13.0"
     ```

If you have built Faiss with GPU support, you can include the "gpu" Cargo feature:

```toml
[dependencies]
"faiss" = { version = "0.13.0", features = ["gpu"] }
```

## Installing with static linking

Alternatively to the above, enable the "static" Cargo feature to let Rust build Faiss for you.
You will still need the dependencies required to build and run Faiss
as described in their [INSTALL.md](https://github.com/Enet4/faiss/blob/c_api_head/INSTALL.md#building-from-source),
namely a compatible C++ compiler and a BLAS implementation.

```toml
[dependencies]
"faiss" = { version = "0.13.0", features = ["static"] }
```

Compiling Faiss with GPU support is also possible.

```toml
[dependencies]
"faiss" = { version = "0.13.0", features = ["static", "gpu"] }
```

## Using

A basic example is seen below. Please check out the [documentation](https://docs.rs/faiss) for more.

```rust
use faiss::{Index, index_factory, MetricType};

let mut index = index_factory(64, "Flat", MetricType::L2)?;
index.add(&my_data)?;

let result = index.search(&my_query, 5)?;
for (i, (l, d)) in result.labels.iter()
    .zip(result.distances.iter())
    .enumerate()
{
    println!("#{}: {} (D={})", i + 1, *l, *d);
}
```

## License and attribution notice

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

This work is not affiliated with Facebook AI Research or the main Faiss software.
