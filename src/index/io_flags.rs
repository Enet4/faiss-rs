//! Module containing the I/O flags.

/// I/O flags used during index reading.
///
/// Note that not all flags are applicable to all index types.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub struct IoFlags(pub(crate) u16);

impl IoFlags {
    /// Load entire index into memory (default behavior)
    pub const MEM_RESIDENT: Self = IoFlags(0x00);
    /// Memory-map index
    pub const MEM_MAP: Self = IoFlags(0x01);
    /// Index is read-only
    pub const READ_ONLY: Self = IoFlags(0x02);
}

impl Default for IoFlags {
    fn default() -> Self {
        IoFlags::MEM_RESIDENT
    }
}

impl std::ops::BitOr for IoFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl From<i32> for IoFlags {
    fn from(n: i32) -> IoFlags {
        IoFlags(n as u16)
    }
}

impl From<IoFlags> for i32 {
    fn from(io_flag: IoFlags) -> i32 {
        io_flag.0 as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_do_bitor() {
        let mmap = IoFlags::MEM_MAP;
        let ro = IoFlags::READ_ONLY;
        assert_eq!(IoFlags(0x03), mmap | ro);
    }

    #[test]
    fn can_coerce_to_i32() {
        let mmap = IoFlags::MEM_MAP;
        assert_eq!(1, mmap.into());
    }
}
