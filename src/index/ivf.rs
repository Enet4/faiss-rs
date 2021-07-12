use faiss_sys::*;

#[derive(Debug, Copy, Clone)]
pub struct IndexIVFStats {
    pub nq: usize,
    pub nlist: usize,
    pub ndis: usize,
    pub nheap_updates: usize,
    pub quantization_time: f64,
    pub search_time: f64,
}

pub fn get_index_ivf_stats() -> Option<IndexIVFStats> {
    unsafe {
        let v = faiss_get_indexIVF_stats();
        if !v.is_null() {
            let v = *v;

            Some(IndexIVFStats {
                nq: v.nq,
                nlist: v.nlist,
                ndis: v.ndis,
                nheap_updates: v.nheap_updates,
                quantization_time: v.quantization_time,
                search_time: v.search_time,
            })
        } else {
            None
        }
    }
}
