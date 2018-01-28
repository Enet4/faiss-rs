extern crate faiss_sys;

macro_rules! faiss_try {
    ($e: expr) => {{
        let c = $e;
        if c != 0 {
            return Err(::error::Error::from_last_error(c));
        }
    }}
}

pub mod error;
pub mod index;
pub mod metric;

pub use index::{Index, index_factory};
