use super::executor::{AbiValue, ArgValue, CatenaMem};

/// Public Catena runtime values accepted at program boundaries.
#[derive(Debug)]
pub enum Value {
    Bool(u8),
    U64(u64),
    Mem(CatenaMem),
}

/// Semantic kinds of public runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Bool,
    U64,
    Mem,
}

impl Value {
    pub(crate) fn kind(&self) -> ValueKind {
        match self {
            Value::Bool(_) => ValueKind::Bool,
            Value::U64(_) => ValueKind::U64,
            Value::Mem(_) => ValueKind::Mem,
        }
    }
}

impl Value {
    pub(crate) fn zeroed(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Bool => Value::Bool(0),
            ValueKind::U64 => Value::U64(0),
            ValueKind::Mem => Value::Mem(CatenaMem {
                data: std::ptr::null_mut(),
                len: 0,
            }),
        }
    }

    pub(crate) fn as_input_arg(&self) -> ArgValue<'_> {
        match self {
            Value::Bool(value) => ArgValue::Val(AbiValue::U8(value)),
            Value::U64(value) => ArgValue::Val(AbiValue::U64(value)),
            Value::Mem(value) => ArgValue::Val(AbiValue::Mem(value)),
        }
    }

    pub(crate) fn as_output_arg(&mut self) -> ArgValue<'_> {
        match self {
            Value::Bool(value) => ArgValue::Out(AbiValue::U8(value)),
            Value::U64(value) => ArgValue::Out(AbiValue::U64(value)),
            Value::Mem(value) => ArgValue::Out(AbiValue::Mem(value)),
        }
    }
}
