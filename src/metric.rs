
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    InnerProduct = 0,
    L2 = 1,
}

impl MetricType {
    pub fn code(&self) -> u32 {
        *self as u32
    }

    pub fn from_code(v: u32) -> Option<Self> {
        match v {
            0 => Some(MetricType::InnerProduct),
            1 => Some(MetricType::L2),
            _ => None,
        }
    }
}