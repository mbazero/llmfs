use std::fmt::Debug;
use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use approx::AbsDiffEq;
use uuid::Uuid;

mod fns {
    use crate::tensor::Tensor;

    pub fn binary_cross_entropy<const D: usize>(
        input: &Tensor<D>,
        target: &Tensor<D>,
    ) -> Tensor<D> {
        const EPS: f64 = 1e-12;
        let input = input.clamp(EPS, 1.0 - EPS);
        let ones = Tensor::ones(input.shape);
        -(target * input.ln() + (&ones - target) * (&ones - &input).ln())
    }
}

#[derive(Debug)]
pub struct Tensor<const D: usize> {
    data: Vec<f64>,
    shape: [usize; D],
    grad_fn: Box<dyn GradFn<D>>,
}

impl<const D: usize> PartialEq for Tensor<D> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl<const D: usize> Tensor<D> {
    pub fn full(shape: [usize; D], value: f64) -> Self {
        let data_len = shape
            .iter()
            .copied()
            .reduce(|a, b| a * b)
            .expect("shape should not be empty");

        Self {
            data: vec![value; data_len],
            shape,
            grad_fn: Box::new(ConstGradFn),
        }
    }

    pub fn zeros(shape: [usize; D]) -> Self {
        Self::full(shape, 0.0)
    }

    pub fn ones(shape: [usize; D]) -> Self {
        Self::full(shape, 1.0)
    }

    pub fn exp(&self) -> Tensor<D> {
        self.unary_op(f64::exp)
    }

    pub fn ln(&self) -> Tensor<D> {
        self.unary_op(f64::ln)
    }

    pub fn log(&self, base: f64) -> Tensor<D> {
        self.unary_op(|x| x.log(base))
    }

    pub fn clamp(&self, min: f64, max: f64) -> Tensor<D> {
        self.unary_op(|x| x.clamp(min, max))
    }

    pub fn sigmoid(&self) -> Tensor<D> {
        Tensor::ones(self.shape) / (Tensor::ones(self.shape) + (-self).exp())
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.data.len() as f64
    }

    fn add(&self, rhs: &Tensor<D>) -> Tensor<D> {
        let data = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(&a, &b)| a + b)
            .collect();

        Self {
            shape: self.shape,
            data,
            grad_fn: Box::new(AddGradFn { a: self, b: rhs }),
        }
    }

    fn unary_op<'a, G: UnaryGradFn<'a, D>>(&'a self, mut op: impl FnMut(f64) -> f64) -> Tensor<D> {
        let grad_fn = G::new(self);
        Tensor {
            shape: self.shape,
            data: self.data.iter().map(|&x| op(x)).collect(),
            grad_fn: Box::new(grad_fn),
        }
    }

    fn binary_op<'a, 'b, G: BinaryGradFn<'a, 'b, D> + 'static>(
        &'a self,
        mut op: impl FnMut(f64, f64) -> f64,
        rhs: &'b Tensor<D>,
    ) -> Tensor<D> {
        assert_eq!(self.shape, rhs.shape, "tensor shapes must be equal");
        Tensor {
            shape: self.shape,
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| op(a, b))
                .collect(),
            grad_fn: Box::new(G::new(self, rhs)),
        }
    }
}

impl<const D: usize> Neg for Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        self.unary_op(f64::neg)
    }
}

impl<const D: usize> Neg for &Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        self.unary_op(f64::neg)
    }
}

impl<const D: usize> Add<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a + b, rhs)
    }
}

impl<const D: usize> Add<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a + b, rhs)
    }
}

impl<const D: usize> Add<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a + b, &rhs)
    }
}

impl<const D: usize> Sub<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a - b, rhs)
    }
}

impl<const D: usize> Sub<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a - b, rhs)
    }
}

impl<const D: usize> Sub<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a - b, &rhs)
    }
}

impl<const D: usize> Mul<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a * b, rhs)
    }
}

impl<const D: usize> Mul<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a * b, rhs)
    }
}

impl<const D: usize> Mul<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a * b, &rhs)
    }
}

impl<const D: usize> Mul<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a * b, &rhs)
    }
}

impl<const D: usize> Div<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a / b, rhs)
    }
}

impl<const D: usize> Div<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: &Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a / b, rhs)
    }
}

impl<const D: usize> Div<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: Tensor<D>) -> Self::Output {
        self.binary_op(|a, b| a / b, &rhs)
    }
}

impl<const D: usize> AbsDiffEq for Tensor<D> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        assert_eq!(self.shape, other.shape, "tensor shapes must be equal");
        self.data
            .iter()
            .zip(&other.data)
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl<const N: usize> From<[f64; N]> for Tensor<1> {
    fn from(value: [f64; N]) -> Self {
        Self {
            shape: [value.len()],
            data: value.into(),
            grad_fn: Box::new(ConstGradFn),
        }
    }
}

pub trait GradFn<const D: usize>: Debug {
    fn grad(&self, diff_var: Tensor<D>) -> Tensor<D>;
}

pub trait UnaryGradFn<'a, const D: usize>: GradFn<D> {
    fn new(a: &'a Tensor<D>) -> Self
    where
        Self: Sized;
}

pub trait BinaryGradFn<'a, 'b, const D: usize>: GradFn<D> {
    fn new(a: &'a Tensor<D>, b: &'b Tensor<D>) -> Self
    where
        Self: Sized;
}

#[derive(Debug, PartialEq, Eq)]
pub struct ConstGradFn;

impl<const D: usize> GradFn<D> for ConstGradFn {
    fn grad(&self, diff_var: Tensor<D>) -> Tensor<D> {
        Tensor::zeros(diff_var.shape)
    }
}

#[derive(Debug, PartialEq)]
pub struct AddGradFn<'a, 'b, const D: usize> {
    a: &'a Tensor<D>,
    b: &'b Tensor<D>,
}

impl<'a, 'b, const D: usize> GradFn<D> for AddGradFn<'a, 'b, D> {
    fn grad(&self, diff_var: Tensor<D>) -> Tensor<D> {
        todo!()
    }
}

impl<'a, 'b, const D: usize> BinaryGradFn<'a, 'b, D> for AddGradFn<'a, 'b, D> {
    fn new(a: &'a Tensor<D>, b: &'b Tensor<D>) -> Self
    where
        Self: Sized,
    {
        Self { a, b }
    }
}

pub enum Op {
    Identity,
    Add,
    Mul,
    Sigmoid,
    BinaryCrossEntropy,
}

pub struct ComputeNode {
    id: Uuid,
    inputs: Vec<Rc<ComputeNode>>,
    op: Op,
}

impl ComputeNode {
    pub fn grad(&self, diff_node: &Rc<ComputeNode>) -> ComputeNode {
        todo!()
    }

    pub fn eval(&self) -> Tensor<1> {
        todo!()
    }
}

impl PartialEq for ComputeNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ComputeNode {}

impl Hash for ComputeNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub fn grad(output: Rc<ComputeNode>, input: Rc<ComputeNode>) -> Tensor<1> {
    let mut q = VecDeque::from([&output]);
    let mut parents = HashMap::new();
    while let Some(cur) = q.pop_front() {
        for child in &cur.inputs {
            parents.insert(child, cur);
            if child == &input {
                break;
            }
        }
    }

    let mut diff_node = &input;
    let mut diff_chain = Vec::new();
    while let Some(&parent) = parents.get(&diff_node) {
        diff_chain.push(parent.grad(diff_node));
        diff_node = parent;
    }

    // diff_chain.iter().map(ComputeNode::eval).sum()
    todo!()
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_basic_autograd() {
        let y = Tensor::from([1.0]);
        let x1 = Tensor::from([1.1]);
        let w1 = Tensor::from([2.2]);
        let b = Tensor::from([0.0]);

        let z = &x1 * &w1 + &b;
        let a = z.sigmoid();

        let loss = fns::binary_cross_entropy(&a, &y);

        assert_abs_diff_eq!(0.0852, loss.mean(), epsilon = 1e-4);
    }
}
