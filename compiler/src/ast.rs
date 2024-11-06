use std::panic::Location;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct VarId(&'static str);

pub mod poly {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum BinOp {
        Add,
        Sub,
        Mul,
        MulPoint,
        Div,
        DivPoint,
    }

    #[derive(Debug, Clone)]
    pub enum ExprNode {
        Bin(BinOp, Expr, Expr),
        Neg(Expr),
        Ntt(Expr),
        MulScalar(scalar::Expr, Expr),
        Id(VarId),
    }


    #[derive(Debug, Clone)]
    pub struct Expr(Rc<ExprNode>, Location<'static>);

    macro_rules! impl_bin_arith {
        ($trait:path, $f:ident, $op:ident) => {
            impl $trait for Expr {
                type Output = Expr;
                #[track_caller]
                fn $f(self, rhs: Expr) -> Self::Output {
                    let loc = Location::caller();
                    Expr(Rc::new(ExprNode::Bin(BinOp::$op, self, rhs)), *loc)
                }
            }
        };
    }

    impl_bin_arith!(std::ops::Add, add, Add);
    impl_bin_arith!(std::ops::Sub, sub, Sub);
    impl_bin_arith!(std::ops::Mul, mul, MulPoint);
    impl_bin_arith!(std::ops::Div, div, DivPoint);

    impl std::ops::Neg for Expr {
        type Output = Expr;
        #[track_caller]
        fn neg(self) -> Self::Output {
            Expr(Rc::new(ExprNode::Neg(self)), *Location::caller())
        }
    }

    #[derive(Debug, Clone)]
    struct CoefRepr(Expr);

    macro_rules! impl_bin_arith_for_coef_repr {
        ($trait:path, $f:ident, $op:ident) => {
            impl $trait for CoefRepr {
                type Output = Expr;
                #[track_caller]
                fn $f(self, rhs: CoefRepr) -> Self::Output {
                    Expr(Rc::new(ExprNode::Bin(BinOp::$op, self.0, rhs.0)), *Location::caller())
                }
            }
        };
    }

    impl_bin_arith_for_coef_repr!(std::ops::Add, add, Add);
    impl_bin_arith_for_coef_repr!(std::ops::Sub, sub, Sub);
    impl_bin_arith_for_coef_repr!(std::ops::Mul, mul, Mul);
    impl_bin_arith_for_coef_repr!(std::ops::Div, div, Div);

    impl Expr {
        pub fn as_coef(self) -> CoefRepr {
            CoefRepr(self)
        }

        #[track_caller]
        pub fn id(s: VarId) -> Self {
            Expr(Rc::new(ExprNode::Id(s)), *Location::caller())
        }

        #[track_caller]
        pub fn ntt(self) -> Expr {
            Expr(Rc::new(ExprNode::Ntt(self)), *Location::caller())
        }
    }

    impl std::ops::Mul<scalar::Expr> for Expr {
        type Output = Expr;
        #[track_caller]
        fn mul(self, rhs: scalar::Expr) -> Self::Output {
            Expr(Rc::new(ExprNode::MulScalar(rhs, self)), *Location::caller())
        }
    }
 
}

pub mod scalar {

    use super::*;

    #[derive(Debug, Clone)]
    pub enum BinOp {
        Add,
        Sub,
        Mul,
        Div,
    }

    #[derive(Debug, Clone)]
    pub enum ExprNode {
        Bin(BinOp, Expr, Expr),
        Neg(Expr),
        IndexCoef(poly::Expr, usize),
        IndexPoint(poly::Expr, usize),
        Id(VarId),
    }

    #[derive(Debug, Clone)]
    pub struct Expr(Rc<ExprNode>, Location<'static>);

    impl std::ops::Add<Expr> for Expr {
        type Output = Expr;
        #[track_caller]
        fn add(self, rhs: Expr) -> Self::Output {
            Expr(Rc::new(ExprNode::Bin(BinOp::Add, self, rhs)), *Location::caller())
        }
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    AssignPoly(VarId, poly::Expr),
    AssignScalar(VarId, scalar::Expr),
    Commit(poly::Expr),
    WriteScalar(VarId),
}
