use std::fmt::Debug;
use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use approx::AbsDiffEq;

pub mod fns {
    use super::*;

    pub fn binary_cross_entropy<const D: usize>(
        input: &Tensor<D>,
        target: &Tensor<D>,
    ) -> Tensor<D> {
        const EPS: f64 = 1e-12;
        let input = input.clamp(EPS, 1.0 - EPS);
        let ones = Tensor::ones(input.data.shape);
        -(target * input.ln() + (&ones - target) * (&ones - &input).ln())
    }
}

#[derive(Debug, PartialEq)]
pub struct TensorData<const D: usize> {
    shape: [usize; D],
    inner: Vec<f64>,
}

impl<const D: usize> TensorData<D> {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn unary_op(&self, op: impl Fn(f64) -> f64) -> Self {
        Self {
            shape: self.shape,
            inner: self.inner.iter().map(|&x| op(x)).collect(),
        }
    }

    fn binary_op(&self, rhs: &TensorData<D>, op: impl Fn(f64, f64) -> f64) -> Self {
        assert_eq!(self.shape, rhs.shape, "tensor shapes must be equal");
        Self {
            shape: self.shape,
            inner: self
                .inner
                .iter()
                .zip(&rhs.inner)
                .map(|(&a, &b)| op(a, b))
                .collect(),
        }
    }
}

impl<const D: usize> AbsDiffEq for TensorData<D> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        assert_eq!(self.shape, other.shape, "tensor shapes must be equal");
        self.inner
            .iter()
            .zip(&other.inner)
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

#[derive(Clone, Debug)]
pub struct Tensor<const D: usize> {
    data: Rc<TensorData<D>>,
    grad_fn: Option<Rc<dyn GradFn<D>>>,
}

impl<const D: usize> PartialEq for Tensor<D> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<const D: usize> Tensor<D> {
    pub fn from_data(data: TensorData<D>) -> Self {
        IdentityGradFn(Rc::new(data)).forward()
    }

    pub fn full(shape: [usize; D], value: f64) -> Self {
        let data_len = shape
            .iter()
            .copied()
            .reduce(|a, b| a * b)
            .expect("shape should not be empty");

        let data = TensorData {
            shape,
            inner: vec![value; data_len],
        };

        Self::from_data(data)
    }

    pub fn zeros(shape: [usize; D]) -> Self {
        Self::full(shape, 0.0)
    }

    pub fn ones(shape: [usize; D]) -> Self {
        Self::full(shape, 1.0)
    }

    pub fn ptr_eq(&self, other: &Tensor<D>) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }

    pub fn neg(&self) -> Tensor<D> {
        grad_fns::neg(self.clone()).forward()
    }

    pub fn exp(&self) -> Tensor<D> {
        grad_fns::exp(self.clone()).forward()
    }

    pub fn ln(&self) -> Tensor<D> {
        grad_fns::ln(self.clone()).forward()
    }

    pub fn clamp(&self, min: f64, max: f64) -> Tensor<D> {
        grad_fns::clamp(self.clone(), min, max).forward()
    }

    pub fn sigmoid(&self) -> Tensor<D> {
        Tensor::ones(self.data.shape) / (Tensor::ones(self.data.shape) + (-self).exp())
    }

    fn add(&self, rhs: &Tensor<D>) -> Tensor<D> {
        grad_fns::add(self.clone(), rhs.clone()).forward()
    }

    fn sub(&self, rhs: &Tensor<D>) -> Tensor<D> {
        grad_fns::add(self.clone(), rhs.neg()).forward()
    }

    fn mul(&self, rhs: &Tensor<D>) -> Tensor<D> {
        grad_fns::mul(self.clone(), rhs.clone()).forward()
    }

    fn div(&self, rhs: &Tensor<D>) -> Tensor<D> {
        grad_fns::div(self.clone(), rhs.clone()).forward()
    }

    pub fn sum(&self) -> f64 {
        self.data.inner.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.data.len() as f64
    }

    pub fn grad(&self, diff_var: &Tensor<D>) -> Tensor<D> {
        self.grad_fn
            .as_ref()
            .expect("tensor should have a grad fn")
            .backward(diff_var)
    }
}

impl<const D: usize> AbsDiffEq for Tensor<D> {
    type Epsilon = <TensorData<D> as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        TensorData::<D>::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.data.abs_diff_eq(&other.data, epsilon)
    }
}

impl<const D: usize> Neg for Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}

impl<const D: usize> Neg for &Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

impl<const D: usize> Add<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

impl<const D: usize> Add<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::add(&self, rhs)
    }
}

impl<const D: usize> Add<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Tensor<D>) -> Self::Output {
        Tensor::add(&self, &rhs)
    }
}

impl<const D: usize> Sub<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

impl<const D: usize> Sub<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::sub(&self, rhs)
    }
}

impl<const D: usize> Sub<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: Tensor<D>) -> Self::Output {
        Tensor::sub(&self, &rhs)
    }
}

impl<const D: usize> Mul<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

impl<const D: usize> Mul<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::mul(&self, rhs)
    }
}

impl<const D: usize> Mul<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Tensor<D>) -> Self::Output {
        Tensor::mul(self, &rhs)
    }
}

impl<const D: usize> Mul<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Tensor<D>) -> Self::Output {
        Tensor::mul(&self, &rhs)
    }
}

impl<const D: usize> Div<&Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::div(self, rhs)
    }
}

impl<const D: usize> Div<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: &Tensor<D>) -> Self::Output {
        Tensor::div(&self, rhs)
    }
}

impl<const D: usize> Div<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: Tensor<D>) -> Self::Output {
        Tensor::div(&self, &rhs)
    }
}

impl<const N: usize> From<[f64; N]> for Tensor<1> {
    fn from(value: [f64; N]) -> Self {
        Self::from_data(TensorData {
            shape: [value.len()],
            inner: value.into(),
        })
    }
}

pub mod grad_fns {
    use super::*;

    pub fn neg<const D: usize>(a: Tensor<D>) -> impl GradFn<D> {
        UnaryGradFn {
            name: "Neg",
            f: a,
            forward_fn: f64::neg,
            backward_fn: |_, df, _| -df,
        }
    }

    pub fn exp<const D: usize>(a: Tensor<D>) -> impl GradFn<D> {
        UnaryGradFn {
            name: "Exp",
            f: a,
            forward_fn: f64::exp,
            backward_fn: |f, df, _| f.exp() * df,
        }
    }

    pub fn ln<const D: usize>(a: Tensor<D>) -> impl GradFn<D> {
        UnaryGradFn {
            name: "Ln",
            f: a,
            forward_fn: f64::ln,
            backward_fn: |f, df, _| df / f,
        }
    }

    pub fn clamp<const D: usize>(a: Tensor<D>, min: f64, max: f64) -> impl GradFn<D> {
        UnaryGradFn {
            name: "Log",
            f: a,
            forward_fn: move |x| x.clamp(min, max),
            backward_fn: move |f, df, _| {
                if (min..=max).contains(&f) { df } else { 0.0 }
            },
        }
    }

    pub fn add<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> impl GradFn<D> {
        BinaryGradFn {
            name: "Add",
            f: a,
            g: b,
            forward_fn: f64::add,
            backward_fn: |_, df, _, dg, _| df + dg,
        }
    }

    pub fn mul<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> impl GradFn<D> {
        BinaryGradFn {
            name: "Mul",
            f: a,
            g: b,
            forward_fn: f64::mul,
            backward_fn: |f, df, g, dg, _| df * g + dg * f,
        }
    }

    pub fn div<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> impl GradFn<D> {
        BinaryGradFn {
            name: "Div",
            f: a,
            g: b,
            forward_fn: f64::div,
            backward_fn: |f, df, g, dg, _| (df * g - dg * f) / (g * g),
        }
    }
}

pub trait GradFn<const D: usize> {
    fn name(&self) -> &str;

    fn forward(self) -> Tensor<D>;

    fn backward(&self, diff_var: &Tensor<D>) -> Tensor<D>;
}

impl<const D: usize> std::fmt::Debug for dyn GradFn<D> + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, PartialEq)]
pub struct IdentityGradFn<const D: usize>(Rc<TensorData<D>>);

impl<const D: usize> GradFn<D> for IdentityGradFn<D> {
    fn name(&self) -> &str {
        "IdentityGradFn"
    }

    fn forward(self) -> Tensor<D> {
        Tensor {
            data: Rc::clone(&self.0),
            grad_fn: Some(Rc::new(self)),
        }
    }

    fn backward(&self, diff_var: &Tensor<D>) -> Tensor<D> {
        if Rc::ptr_eq(&self.0, &diff_var.data) {
            Tensor::ones(self.0.shape)
        } else {
            Tensor::zeros(self.0.shape)
        }
    }
}

#[derive(PartialEq)]
pub struct UnaryGradFn<const D: usize, F, B>
where
    F: Fn(f64) -> f64,
    B: Fn(f64, f64, f64) -> f64,
{
    name: &'static str,
    f: Tensor<D>,
    forward_fn: F,
    backward_fn: B,
}

impl<const D: usize, F, B> GradFn<D> for UnaryGradFn<D, F, B>
where
    F: Fn(f64) -> f64 + Copy + 'static,
    B: Fn(f64, f64, f64) -> f64 + Copy + 'static,
{
    fn name(&self) -> &str {
        self.name
    }

    fn forward(self) -> Tensor<D> {
        Tensor {
            data: Rc::new(self.f.data.unary_op(self.forward_fn)),
            grad_fn: Some(Rc::new(self)),
        }
    }

    fn backward(&self, diff_var: &Tensor<D>) -> Tensor<D> {
        let df = self.f.grad(diff_var);

        let data = TensorData {
            shape: self.f.data.shape,
            inner: (0..self.f.data.len())
                .map(|i| {
                    (self.backward_fn)(
                        self.f.data.inner[i],
                        df.data.inner[i],
                        diff_var.data.inner[i],
                    )
                })
                .collect(),
        };

        Tensor {
            data: data.into(),
            grad_fn: None,
        }
    }
}

#[derive(PartialEq)]
pub struct BinaryGradFn<const D: usize, F, B>
where
    F: Fn(f64, f64) -> f64,
    B: Fn(f64, f64, f64, f64, f64) -> f64,
{
    name: &'static str,
    f: Tensor<D>,
    g: Tensor<D>,
    forward_fn: F,
    backward_fn: B,
}

impl<const D: usize, F, B> GradFn<D> for BinaryGradFn<D, F, B>
where
    F: Fn(f64, f64) -> f64 + Copy + 'static,
    B: Fn(f64, f64, f64, f64, f64) -> f64 + Copy + 'static,
{
    fn name(&self) -> &str {
        self.name
    }

    fn forward(self) -> Tensor<D> {
        Tensor {
            data: Rc::new(self.f.data.binary_op(&self.g.data, self.forward_fn)),
            grad_fn: Some(Rc::new(self)),
        }
    }

    fn backward(&self, diff_var: &Tensor<D>) -> Tensor<D> {
        let df = self.f.grad(diff_var);
        let dg = self.g.grad(diff_var);

        let data = TensorData {
            shape: self.f.data.shape,
            inner: (0..self.f.data.len())
                .map(|i| {
                    (self.backward_fn)(
                        self.f.data.inner[i],
                        df.data.inner[i],
                        self.g.data.inner[i],
                        dg.data.inner[i],
                        diff_var.data.inner[i],
                    )
                })
                .collect(),
        };

        Tensor {
            data: data.into(),
            grad_fn: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_add_mul_autograd() {
        let m = Tensor::from([1.1]);
        let x = Tensor::from([2.2]);
        let b = Tensor::from([3.3]);

        let y = &m * &x + &b;
        let grad_y_x = y.grad(&x);

        assert_abs_diff_eq!(Tensor::from([5.72]), y, epsilon = 1e-4);
        assert_abs_diff_eq!(Tensor::from([1.1]), grad_y_x, epsilon = 1e-4);
    }

    #[test]
    fn test_bce_autograd() {
        let y = Tensor::from([1.0]);
        let x1 = Tensor::from([1.1]);
        let w1 = Tensor::from([2.2]);
        let b = Tensor::from([0.0]);

        let z = &x1 * &w1 + &b;
        let a = z.sigmoid();

        let loss = fns::binary_cross_entropy(&a, &y);

        assert_abs_diff_eq!(Tensor::from([0.0852]), loss, epsilon = 1e-4);

        assert_abs_diff_eq!(Tensor::from([1.0]), w1.grad(&w1), epsilon = 1e-4);
        assert_abs_diff_eq!(Tensor::from([1.1]), z.grad(&w1), epsilon = 1e-4);
        assert_abs_diff_eq!(Tensor::from([0.0825]), a.grad(&w1), epsilon = 1e-4);
        assert_abs_diff_eq!(Tensor::from([-0.0898]), loss.grad(&w1), epsilon = 1e-4);

        assert_abs_diff_eq!(Tensor::from([1.0]), b.grad(&b), epsilon = 1e-4);
        assert_abs_diff_eq!(Tensor::from([1.0]), z.grad(&b), epsilon = 1e-4);
        assert_abs_diff_eq!(Tensor::from([0.075]), a.grad(&b), epsilon = 1e-4);
        assert_abs_diff_eq!(Tensor::from([-0.0817]), loss.grad(&b), epsilon = 1e-4);
    }
}
