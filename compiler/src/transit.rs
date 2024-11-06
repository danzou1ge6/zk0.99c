//! Common data structures for Transit IR's

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::digraph::internal::{Digraph, Successors};
use crate::heap::{Heap, UsizeId};
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct SourceInfo<'s> {
    _marker: PhantomData<&'s str>,
}

/// Computation Graph of a Transit IR function.
/// [`V`]: vertex
/// [`I`]: vertex ID
#[derive(Debug, Clone)]
pub struct Cg<I, V> {
    inputs: Vec<I>,
    outputs: Vec<I>,
    g: Digraph<I, V>,
}

#[derive(Debug, Clone)]
pub enum ArithBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub enum ArithUnrOp {
    Neg,
    Inv,
}

/// Scalar-Polynomial operator
#[derive(Debug, Clone)]
pub enum SpOp {
    Add,
    Sub,
    SubBy,
    Mul,
    Div,
    DivBy,
    Eval,
}

/// Unary polynomial operator
#[derive(Debug, Clone)]
pub enum POp {
    Neg,
    Inv,
    Roatate(isize),
}

mod op_template {

    /// Binary operator.
    /// [`P`]: polynomial-Polynomial opertor
    #[derive(Debug, Clone)]
    pub enum BinOp<Pp, Ss, Sp> {
        Pp(Pp),
        Ss(Ss),
        Sp(Sp),
    }

    /// Unary operator.
    /// [`Po`]: polynomial unary operator
    #[derive(Debug, Clone)]
    pub enum UnrOp<Po, So> {
        P(Po),
        S(So),
    }
}

pub type BinOp = op_template::BinOp<ArithBinOp, ArithBinOp, SpOp>;
pub type UnrOp = op_template::UnrOp<POp, ArithUnrOp>;

/// Kind-specific data of expressions.
#[derive(Debug, Clone)]
pub enum Arith<I> {
    Bin(BinOp, I, I),
    Unr(UnrOp, I),
}

#[derive(Debug, Clone)]
pub struct Vertex<N, T, S>(N, T, S);

impl<N, T, S> Vertex<N, T, S> {
    pub fn new(v_node: N, v_typ: T, v_src: S) -> Self {
        Self(v_node, v_typ, v_src)
    }
    pub fn node(&self) -> &N {
        &self.0
    }
    pub fn typ(&self) -> &T {
        &self.1
    }
    pub fn src(&self) -> &S {
        &self.2
    }
    pub fn node_mut(&mut self) -> &mut N {
        &mut self.0
    }
    pub fn typ_mut(&mut self) -> &mut T {
        &mut self.1
    }
    pub fn src_mut(&mut self) -> &mut S {
        &mut self.2
    }
}

pub mod type1;
pub mod type2;
