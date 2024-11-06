//! Data structures for Stage 2 Tree Transit IR.
//! This type of IR has NTT automatically inserted,
//! therefore polynomial operations are transformed into vector expressions.
//! Also, vertices of the computation graph still contain expression trees.

use crate::{
    digraph, heap,
    transit::{self, BinOp, SourceInfo, UnrOp},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyRepr {
    Coef(usize),
    Lagrange(usize),
    ExtendedLagrange(usize),
}

#[derive(Debug, Clone)]
pub enum Typ {
    Poly(PolyRepr),
    Scalar,
}

crate::define_usize_id!(ExprId);

impl ExprId {
    pub(super) fn from_type1(x: transit::type1::ExprId) -> Self {
        Self(x.into())
    }
}

pub type Arith = transit::Arith<ExprId>;

#[derive(Debug, Clone)]
pub enum VertexNode {
    Arith(Arith),
    Entry,
    Return,
    /// A small scalar. To accomodate big scalars, use global constants.
    LiteralScalar(usize),
    /// Convert a local from one representation to another
    Ntt {
        s: ExprId,
        to: PolyRepr,
        from: PolyRepr,
    },
    Interplote {
        xs: Vec<ExprId>,
        ys: Vec<ExprId>,
    },
    AssmblePoly(usize, Vec<ExprId>),
}

pub type Vertex<'s> = transit::Vertex<VertexNode, Typ, SourceInfo<'s>>;

impl<'s> digraph::internal::Predecessors<ExprId> for Vertex<'s> {
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> Box<dyn Iterator<Item = ExprId> + 'a> {
        self.uses()
    }
}

impl<'s> Vertex<'s> {
    pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = ExprId> + 'a> {
        use VertexNode::*;
        match self.node() {
            Arith(transit::Arith::Bin(_, lhs, rhs)) => Box::new([*lhs, *rhs].into_iter()),
            Arith(transit::Arith::Unr(_, x)) => Box::new([*x].into_iter()),
            Ntt { s, .. } => Box::new([*s].into_iter()),
            Interplote { xs, ys } => Box::new(xs.iter().copied().chain(ys.iter().copied())),

            AssmblePoly(_, es) => Box::new(es.iter().copied()),
            _ => Box::new([].into_iter()),
        }
    }
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut ExprId> + 'a> {
        use VertexNode::*;
        match self.node_mut() {
            Arith(transit::Arith::Bin(_, lhs, rhs)) => Box::new([lhs, rhs].into_iter()),
            Arith(transit::Arith::Unr(_, x)) => Box::new([x].into_iter()),
            Ntt { s, .. } => Box::new([s].into_iter()),
            Interplote { xs, ys } => Box::new(xs.iter_mut().chain(ys.iter_mut())),

            AssmblePoly(_, es) => Box::new(es.iter_mut()),
            _ => Box::new([].into_iter()),
        }
    }
    pub fn modify_locals(&mut self, f: &mut impl FnMut(ExprId) -> ExprId) {
        self.uses_mut().for_each(|i| *i = f(*i));
    }
}

/// Invariants are same as those of [`tree::tree1::Cg`]
pub type Cg<'s> = transit::Cg<ExprId, Vertex<'s>>;
