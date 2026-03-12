use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

// Client → Server messages
pub enum Message {}

#[derive(Clone, Debug, PartialEq)]
pub struct Type;

#[derive(Clone, Debug)]
pub enum Operation {
    LoadArray(usize), // load a number of bytes as a constant array (of bytes)
    CatenaProgram(CatenaCode),
}

pub type Trace = OpenHypergraph<Type, Operation>;

// values
#[derive(Clone, Debug)]
pub struct Value;

#[derive(Clone, Debug)]
pub struct State {
    trace: Trace,
    state: Vec<Value>,
}

#[derive(Clone, Debug, Error)]
pub enum StepError {
    #[error("Type error")]
    ComposeTypeError,
}

impl State {
    pub fn new() -> State {
        Self {
            trace: OpenHypergraph::empty(),
            state: vec![],
        }
    }

    pub fn step(&mut self, next: Trace) -> Result<(), StepError> {
        use open_hypergraphs::category::Arrow;
        // TODO: do this mutably on 'trace' - add mutable composition operations to lax OpenHypergraph
        self.trace = self
            .trace
            .compose(&next)
            .ok_or(StepError::ComposeTypeError)?;

        // compute the next values eagerly
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// temp: deleteme

// TODO: what is this?
#[derive(Clone, Debug)]
pub struct CatenaCode;
