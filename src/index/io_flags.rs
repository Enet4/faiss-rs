//! Module containing the io flags.

/// Io Flags used during index reading, not all flags applicable to all indices
// This is a set of constants rather than enum so that bitwise operations exist
pub mod io_flag {
    /// Load entire index into memory (default behavior)
    pub const MEM_RESIDENT: u8 = 0x00;
    /// Memory-map index
    pub const MEM_MAP: u8 = 0x01;
    /// Index is read-only
    pub const READ_ONLY: u8 = 0x02;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_do_bitor() {
        let mmap = io_flag::MEM_MAP;
        let ro = io_flag::READ_ONLY;
        assert_eq!(3, mmap | ro);
    }
}
