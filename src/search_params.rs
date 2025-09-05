// //! Abstract Faiss SearchParameters and SearchParametersIVF

use faiss_sys::*;
use std::{mem, ptr};
use crate::error::{Error, Result};
use crate::faiss_try;
use crate::selector::IdSelector;

pub type SearchParameters = SearchParametersImpl;


#[derive(Debug)]
pub struct SearchParametersImpl {
    inner: *mut FaissSearchParameters
}

impl SearchParametersImpl {
    pub fn inner_ptr(&self) -> *mut FaissSearchParameters {
        self.inner
    }
    /// Create new search parameters from IdSelector.
    pub fn new(selector: IdSelector) -> Result<Self> {

        let mut p_sp = ptr::null_mut();
        let p_sel = selector.inner_ptr();
        unsafe {
            faiss_try(faiss_SearchParameters_new(
                &mut p_sp, 
                p_sel,
            ))?;
        }
        mem::forget(selector);
        Ok(SearchParametersImpl { inner: p_sp as *mut _ })
    }
}

impl Drop for SearchParametersImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_SearchParameters_free(self.inner);
        }
    }
}

unsafe impl Send for SearchParametersImpl {}
unsafe impl Sync for SearchParametersImpl {}


pub type SearchParametersIVF = SearchParametersIVFImpl;

#[derive(Debug)]
pub struct SearchParametersIVFImpl {
    inner: *mut FaissSearchParametersIVF
}

impl SearchParametersIVFImpl {

    /// Create new empty search parameters.
    pub fn new() -> Result<Self> {
        let mut p_sp = ptr::null_mut();

        unsafe { 
            faiss_try(faiss_SearchParametersIVF_new(
                &mut p_sp
            ))?;
        }

        Ok(SearchParametersIVFImpl { inner: p_sp as *mut _ })
    }

    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { faiss_SearchParametersIVF_set_nprobe(self.inner, nprobe); }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { faiss_SearchParametersIVF_nprobe(self.inner) }
    }

    pub fn set_max_codes(&mut self, max_codes: usize) {
        unsafe { faiss_SearchParametersIVF_set_max_codes(self.inner, max_codes); }
    }

    pub fn max_codes(&self) -> usize {
        unsafe { faiss_SearchParametersIVF_max_codes(self.inner) }
    }

    /// Create a new SearchParameterIVF using the given selector, and other params.
    pub fn new_with(
        selector: IdSelector,
        nprobe: usize,
        max_codes: usize
    ) -> Result<Self> {

        let mut p_sp = ptr::null_mut();
        let p_sel = selector.inner_ptr();

        unsafe { 
            faiss_try(faiss_SearchParametersIVF_new_with(
                &mut p_sp,
                p_sel,
                nprobe,
                max_codes
            ))?;
        }

        mem::forget(selector);
        Ok(SearchParametersIVFImpl { inner: p_sp as *mut _ })
    }

    pub fn upcast(self) -> SearchParametersImpl {
        let inner_ptr = self.inner;
        mem::forget(self);
        SearchParametersImpl { inner: inner_ptr }
    }

    pub fn inner_ptr(&self) -> *mut FaissSearchParameters {
        self.inner
    }

}

impl Drop for SearchParametersIVFImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_SearchParametersIVF_free(self.inner);
        }
    }
}

unsafe impl Send for SearchParametersIVFImpl {}
unsafe impl Sync for SearchParametersIVFImpl {}

impl SearchParameters {
    pub fn into_search_parameters_ivf(self) -> Result<SearchParametersIVF> {
        unsafe {
            let new_inner = faiss_SearchParametersIVF_cast(self.inner);
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(SearchParametersIVFImpl { inner: new_inner })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Idx;

    #[test]
    fn search_params_from_id_selector() {
        let selector = IdSelector::range(Idx::new(0), Idx::new(10)).unwrap();
        let _search_params = SearchParametersImpl::new(selector).unwrap();
    }

    #[test]
    fn search_params_ivf() {
        let mut params = SearchParametersIVFImpl::new().unwrap();
        assert_eq!(params.nprobe(), 1);
        assert_eq!(params.max_codes(), 0);

        params.set_max_codes(10);
        params.set_nprobe(10);
        assert_eq!(params.nprobe(), 10);
        assert_eq!(params.max_codes(), 10);
    }

    #[test]
    fn search_params_ivf_with_selector() {
        let selector = IdSelector::range(Idx::new(0), Idx::new(10)).unwrap();
        let params = SearchParametersIVFImpl::new_with(
            selector,
            1,
            1
        ).unwrap();
        assert_eq!(params.max_codes(), 1);
        assert_eq!(params.nprobe(), 1);
    }

    #[test]
    fn search_params_ivf_cast() {
        let selector = IdSelector::range(Idx::new(0), Idx::new(10)).unwrap();
        let params_ivf = SearchParametersIVFImpl::new_with(
            selector,
            1,
            1
        ).unwrap();
        assert_eq!(params_ivf.max_codes(), 1);
        assert_eq!(params_ivf.nprobe(), 1);

        let params = params_ivf.upcast();
        let params_ivf = params.into_search_parameters_ivf().unwrap();
        assert_eq!(params_ivf.max_codes(), 1);
        assert_eq!(params_ivf.nprobe(), 1);
    }
}