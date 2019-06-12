//! Module containing the metric type.

/// Enumerate type describing the type of metric assumed by an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Inner product, also called cosine distance
    InnerProduct = 0,
    /// Euclidean L2-distance
    L2 = 1,
}

impl MetricType {
    /// Obtain the native code which identifies this metric type.
    pub fn code(self) -> u32 {
        self as u32
    }

    /// Obtain a metric type value from the native code.
    pub fn from_code(v: u32) -> Option<Self> {
        match v {
            0 => Some(MetricType::InnerProduct),
            1 => Some(MetricType::L2),
            _ => None,
        }
    }
}
