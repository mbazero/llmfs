use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
    iter::Sum,
    ops::Add,
    rc::Rc,
};

use uuid::Uuid;

pub struct Tensor<const D: usize> {}

impl<const D: usize> Tensor<D> {
    pub fn zero() -> Self {
        todo!()
    }
}

impl<const D: usize> Add<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Tensor<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Tensor<D>) -> Self::Output {
        &self + rhs
    }
}

impl<const D: usize> Add<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Tensor<D>) -> Self::Output {
        &self + &rhs
    }
}

impl<const D: usize> Sum for Tensor<D> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
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

    pub fn eval(&self) -> Tensor {
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

pub fn grad(output: Rc<ComputeNode>, input: Rc<ComputeNode>) -> Tensor {
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

    diff_chain.iter().map(ComputeNode::eval).sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic_autograd() {}
}
