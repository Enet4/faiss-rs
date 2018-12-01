macro_rules! faiss_try {
    ($e:expr) => {{
        let c = $e;
        if c != 0 {
            return Err(::error::NativeError::from_last_error(c).into());
        }
    }};
}

/// A macro which provides a native index implementation to the given type.
macro_rules! impl_native_index {
    ($t:ty) => {
        impl ::index::Index for $t {
            fn is_trained(&self) -> bool {
                unsafe { faiss_Index_is_trained(self.inner_ptr()) != 0 }
            }

            fn ntotal(&self) -> u64 {
                unsafe { faiss_Index_ntotal(self.inner_ptr()) as u64 }
            }

            fn d(&self) -> u32 {
                unsafe { faiss_Index_d(self.inner_ptr()) as u32 }
            }

            fn metric_type(&self) -> ::metric::MetricType {
                unsafe {
                    ::metric::MetricType::from_code(faiss_Index_metric_type(self.inner_ptr()) as u32).unwrap()
                }
            }

            fn add(&mut self, x: &[f32]) -> Result<()> {
                unsafe {
                    let n = x.len() / self.d() as usize;
                    faiss_try!(faiss_Index_add(self.inner_ptr(), n as i64, x.as_ptr()));
                    Ok(())
                }
            }

            fn add_with_ids(&mut self, x: &[f32], xids: &[::index::Idx]) -> Result<()> {
                unsafe {
                    let n = x.len() / self.d() as usize;
                    faiss_try!(faiss_Index_add_with_ids(
                        self.inner_ptr(),
                        n as i64,
                        x.as_ptr(),
                        xids.as_ptr()
                    ));
                    Ok(())
                }
            }
            fn train(&mut self, x: &[f32]) -> Result<()> {
                unsafe {
                    let n = x.len() / self.d() as usize;
                    faiss_try!(faiss_Index_train(self.inner_ptr(), n as i64, x.as_ptr()));
                    Ok(())
                }
            }
            fn assign(&mut self, query: &[f32], k: usize) -> Result<::index::AssignSearchResult> {
                unsafe {
                    let nq = query.len() / self.d() as usize;
                    let mut out_labels = vec![0 as ::index::Idx; k * nq];
                    faiss_try!(faiss_Index_assign(
                        self.inner_ptr(),
                        nq as idx_t,
                        query.as_ptr(),
                        out_labels.as_mut_ptr(),
                        k as i64
                    ));
                    Ok(::index::AssignSearchResult { labels: out_labels })
                }
            }
            fn search(&mut self, query: &[f32], k: usize) -> Result<::index::SearchResult> {
                unsafe {
                    let nq = query.len() / self.d() as usize;
                    let mut distances = vec![0_f32; k * nq];
                    let mut labels = vec![0 as ::index::Idx; k * nq];
                    faiss_try!(faiss_Index_search(
                        self.inner_ptr(),
                        nq as idx_t,
                        query.as_ptr(),
                        k as idx_t,
                        distances.as_mut_ptr(),
                        labels.as_mut_ptr()
                    ));
                    Ok(::index::SearchResult { distances, labels })
                }
            }
            fn range_search(&mut self, query: &[f32], radius: f32) -> Result<::index::RangeSearchResult> {
                unsafe {
                    let nq = (query.len() / self.d() as usize) as idx_t;
                    let mut p_res: *mut FaissRangeSearchResult = ::std::ptr::null_mut();
                    faiss_try!(faiss_RangeSearchResult_new(&mut p_res, nq));
                    faiss_try!(faiss_Index_range_search(
                        self.inner_ptr(),
                        nq,
                        query.as_ptr(),
                        radius,
                        p_res
                    ));
                    Ok(::index::RangeSearchResult { inner: p_res })
                }
            }

            fn reset(&mut self) -> Result<()> {
                unsafe {
                    faiss_try!(faiss_Index_reset(self.inner_ptr()));
                    Ok(())
                }
            }

            fn remove_ids(&mut self, sel: &[faiss_sys::FaissIDSelector_H]) -> Result<()> {
                unsafe {
                    let mut n_removed = 0;
                    faiss_try!(faiss_Index_remove_ids(
                        self.inner_ptr(),
                        sel.as_ptr(),
                        &mut n_removed
                    ));
                    Ok(())
                }
            }
        }
    };
}

/// A macro which provides a Clone implementation to native index types.
macro_rules! impl_native_index_clone {
    ($t:ty) => {
        impl $t {
            /// Create an independent clone of this index.
            ///
            /// # Errors
            ///
            /// May result in a native error if the clone operation is not
            /// supported for the internal type of index.
            pub fn try_clone(&self) -> Result<Self> {
                unsafe {
                    let mut new_index_ptr = ::std::ptr::null_mut();
                    faiss_try!(faiss_clone_index(self.inner_ptr(), &mut new_index_ptr));
                    Ok(::index::FromInnerPtr::from_inner_ptr(new_index_ptr))
                }
            }
        }
    };
}
