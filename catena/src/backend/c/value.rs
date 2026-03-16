/// Public catena runtime values for the C backend.
#[derive(Debug)]
pub enum Value {
    Extent(usize),
    Index(usize),
    F32(f32),
}

/// Semantic kinds of public runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Extent,
    Index,
    F32,
}

impl Value {
    pub(crate) fn kind(&self) -> ValueKind {
        match self {
            Value::Extent(_) => ValueKind::Extent,
            Value::Index(_) => ValueKind::Index,
            Value::F32(_) => ValueKind::F32,
        }
    }
}
