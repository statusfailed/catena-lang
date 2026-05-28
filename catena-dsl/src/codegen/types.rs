use hexpr::Operation;
use metacat::tree::Tree;

const VALUE_TYPES: &[&str] = &["val", "value"];

/// Map a Catena object type to the type string stored on a `StructuredProgram` parameter.
///
/// Example:
/// `bool` becomes `uint8_t`
///
/// Example:
/// output `bool` becomes `uint8_t*`
pub fn structured_param_type(obj: &Tree<(), Operation>, by_pointer: bool) -> Option<String> {
    let base = scalar_type(obj)?;
    if by_pointer {
        Some(format!("{base}*"))
    } else {
        Some(base)
    }
}

/// Render a Catena object type as a full GPU parameter declaration.
///
/// Example:
/// `bool` with name `x0` becomes `uint8_t x0`
///
/// Example:
/// output `bool` with name `out_x1` becomes `uint8_t *out_x1`
pub fn gpu_param_decl(obj: &Tree<(), Operation>, name: &str, by_pointer: bool) -> Option<String> {
    if by_pointer {
        scalar_type(obj).map(|ty| format!("{ty} *{name}"))
    } else {
        declaration(obj, name)
    }
}

/// Render a Catena object type as a full GPU local declaration.
///
/// Example:
/// `bool` with name `x3` becomes `uint8_t x3`
///
/// Example:
/// `(bool -> bool)` with name `f` becomes `void (*f)(uint8_t arg0, uint8_t *out0)`
pub fn gpu_local_decl(obj: &Tree<(), Operation>, name: &str) -> Option<String> {
    declaration(obj, name)
}

pub fn gpu_value_type(obj: &Tree<(), Operation>) -> Option<String> {
    match obj {
        Tree::Node(op, 0, children) if op.as_str() == "bool" && children.is_empty() => {
            Some("uint8_t".to_string())
        }
        Tree::Node(op, 0, children) if op.as_str() == "u64" && children.is_empty() => {
            Some("uint64_t".to_string())
        }
        Tree::Node(op, 0, children) if op.as_str() == "gpu.3d" && children.is_empty() => {
            Some("catena_dim3_t".to_string())
        }
        Tree::Node(op, 0, children)
            if op.as_str() == "gpu.launch_params" && children.is_empty() =>
        {
            Some("catena_launch_params_t".to_string())
        }
        Tree::Node(op, 0, children) if op.as_str() == "gpu.env" && children.is_empty() => {
            Some("catena_gpu_env_t".to_string())
        }
        Tree::Node(op, 0, children) if op.as_str() == "gpu.state" && children.is_empty() => {
            Some("catena_gpu_state_t".to_string())
        }
        Tree::Node(op, 0, children) if op.as_str() == "gpu.buf" => {
            let [_element] = children.as_slice() else {
                return None;
            };
            Some("catena_gpu_buf_t".to_string())
        }
        _ => None,
    }
}

pub fn gpu_buffer_element_type(obj: &Tree<(), Operation>) -> Option<String> {
    let Tree::Node(op, 0, children) = obj else {
        return None;
    };
    if op.as_str() != "gpu.buf" {
        return None;
    }
    let [element] = children.as_slice() else {
        return None;
    };
    runtime_inner(element)
        .and_then(gpu_value_type)
        .or_else(|| gpu_value_type(element))
}

fn scalar_type(obj: &Tree<(), Operation>) -> Option<String> {
    if let Some(control) = gpu_value_type(obj) {
        return Some(control);
    }

    match runtime_inner(obj)? {
        Tree::Node(op, 0, children) if op.as_str() == "1" && children.is_empty() => {
            Some("catena_unit_t".to_string())
        }
        Tree::Node(op, 0, children) if op.as_str() == "->" => {
            let [source, target] = children.as_slice() else {
                return None;
            };
            Some(format!(
                "fn_{}_to_{}",
                mangle_object(source),
                mangle_object(target)
            ))
        }
        inner => gpu_value_type(inner),
    }
}

fn declaration(obj: &Tree<(), Operation>, name: &str) -> Option<String> {
    if let Some(control) = gpu_value_type(obj) {
        return Some(format!("{control} {name}"));
    }

    match runtime_inner(obj)? {
        Tree::Node(op, 0, children) if op.as_str() == "1" && children.is_empty() => {
            Some(format!("catena_unit_t {name}"))
        }
        Tree::Node(op, 0, children) if op.as_str() == "bool" && children.is_empty() => {
            Some(format!("uint8_t {name}"))
        }
        Tree::Node(op, 0, children) if op.as_str() == "->" => {
            let [source, target] = children.as_slice() else {
                return None;
            };
            let params = gpu_fn_param_list(source, target)?;
            Some(format!("void (*{name})({params})"))
        }
        _ => None,
    }
}

fn gpu_fn_param_list(source: &Tree<(), Operation>, target: &Tree<(), Operation>) -> Option<String> {
    let mut parts = runtime_components(source)
        .into_iter()
        .enumerate()
        .map(|(index, arg)| declaration(arg, &format!("arg{index}")))
        .collect::<Option<Vec<_>>>()?;

    let source_len = parts.len();
    let mut outputs = runtime_components(target)
        .into_iter()
        .enumerate()
        .map(|(index, arg)| gpu_param_decl(arg, &format!("out{}", source_len + index), true))
        .collect::<Option<Vec<_>>>()?;
    parts.append(&mut outputs);

    if parts.is_empty() {
        Some("void".to_string())
    } else {
        Some(parts.join(", "))
    }
}

fn runtime_components<'a>(obj: &'a Tree<(), Operation>) -> Vec<&'a Tree<(), Operation>> {
    match obj {
        Tree::Node(op, 0, children) if op.as_str() == "*" => {
            children.iter().flat_map(runtime_components).collect()
        }
        other if runtime_inner(other).is_some() || gpu_value_type(other).is_some() => vec![other],
        _ => Vec::new(),
    }
}

fn mangle_object(obj: &Tree<(), Operation>) -> String {
    match obj {
        Tree::Empty => "empty".to_string(),
        Tree::Leaf(index, _) => format!("x{index}"),
        Tree::Node(op, port, children) => {
            let mut out = sanitize_ident(op.as_str());
            if *port != 0 {
                out.push('_');
                out.push_str(&port.to_string());
            }
            for child in children {
                out.push('_');
                out.push_str(&mangle_object(child));
            }
            out
        }
    }
}

fn runtime_inner(obj: &Tree<(), Operation>) -> Option<&Tree<(), Operation>> {
    let Tree::Node(op, 0, children) = obj else {
        return None;
    };
    if !VALUE_TYPES.contains(&op.as_str()) {
        return None;
    }
    let [inner] = children.as_slice() else {
        return None;
    };
    Some(inner)
}

fn sanitize_ident(value: &str) -> String {
    let mut ident = value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.is_empty() {
        ident.push('_');
    }
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_type() -> Tree<(), Operation> {
        Tree::Node("1".parse().unwrap(), 0, vec![])
    }

    fn bool_type() -> Tree<(), Operation> {
        Tree::Node("bool".parse().unwrap(), 0, vec![])
    }

    fn val_type(inner: Tree<(), Operation>) -> Tree<(), Operation> {
        Tree::Node("val".parse().unwrap(), 0, vec![inner])
    }

    fn fn_ptr_type(
        source: Tree<(), Operation>,
        target: Tree<(), Operation>,
    ) -> Tree<(), Operation> {
        Tree::Node("->".parse().unwrap(), 0, vec![source, target])
    }

    #[test]
    fn structured_param_type_renders_function_pointer_name() {
        let ty = val_type(fn_ptr_type(val_type(bool_type()), val_type(bool_type())));
        assert_eq!(
            structured_param_type(&ty, false).as_deref(),
            Some("fn_val_bool_to_val_bool")
        );
        assert_eq!(
            structured_param_type(&ty, true).as_deref(),
            Some("fn_val_bool_to_val_bool*")
        );
    }

    #[test]
    fn unit_type_renders_as_concrete_gpu_type() {
        let ty = val_type(unit_type());
        assert_eq!(
            structured_param_type(&ty, false).as_deref(),
            Some("catena_unit_t")
        );
        assert_eq!(
            gpu_param_decl(&ty, "u", false).as_deref(),
            Some("catena_unit_t u")
        );
        assert_eq!(
            gpu_local_decl(&ty, "tmp").as_deref(),
            Some("catena_unit_t tmp")
        );
    }

    #[test]
    fn gpu_declarations_render_function_pointer_types() {
        let ty = val_type(fn_ptr_type(val_type(bool_type()), val_type(bool_type())));
        assert_eq!(
            gpu_param_decl(&ty, "f", false).as_deref(),
            Some("void (*f)(uint8_t arg0, uint8_t *out1)")
        );
        assert_eq!(
            gpu_local_decl(&ty, "tmp").as_deref(),
            Some("void (*tmp)(uint8_t arg0, uint8_t *out1)")
        );
    }

    #[test]
    fn gpu_declarations_skip_non_runtime_function_inputs() {
        let ty = val_type(fn_ptr_type(Tree::Leaf(0, ()), val_type(bool_type())));
        assert_eq!(
            gpu_param_decl(&ty, "f", false).as_deref(),
            Some("void (*f)(uint8_t *out0)")
        );
    }
}
