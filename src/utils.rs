/// L2-renormalize a set of vector. Nothing done if the vector is 0-normed
pub fn fvec_renorm_l2(d: usize, nx:usize, fvec: &mut[f32]) {
    unsafe {
        faiss_sys::faiss_fvec_renorm_L2(d, nx, fvec.as_mut_ptr())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    const D: u32 = 8;

    #[test]
    fn check_fvec_renorm_l2_01() {
        let mut some_data = vec![
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];

        fvec_renorm_l2(D as usize, 5, &mut some_data);
    }
}