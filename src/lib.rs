mod graph;
mod optimize;

use nalgebra as na;

pub type Param = na::DVector<f64>;
pub type SolveCoeff = na::DMatrix<f64>;
pub type Residual = na::DVector<f64>;
pub type Jacobian = na::DMatrix<f64>;

pub trait LMProblem {
    fn solve(&self, x: &na::DMatrix<f64>, y: &na::DVector<f64>) -> na::DVector<f64>;
    fn residual(&self, x: &na::DVector<f64>) -> na::DVector<f64>; 
    fn jacobian(&self, x: &na::DVector<f64>) -> na::DMatrix<f64>;
}

pub struct LM {
    problem: Box<dyn LMProblem>,
    tau: f64,
    eps1: f64,
    eps2: f64,
    eps3: f64,
    v: f64,
    max_iter: usize,
}

impl LM {
    pub fn new(
        problem: Box<dyn LMProblem>, 
        tau: f64, 
        eps1: f64,
        eps2: f64,
        eps3: f64,
        v: f64,
        max_iter: usize, 
    ) -> Self {
        Self {
            problem,
            tau,
            eps1,
            eps2,
            eps3,
            v,
            max_iter,
        }
    }

    pub fn default(problem: Box<dyn LMProblem>) -> Self {
        Self::new(problem, 1e-3, 1e-15, 1e-15, 1e-15, 2.0, 100)
    }

    pub fn optimize(&self, x0: &na::DVector<f64>) -> na::DVector<f64> {
        let mut v = self.v;
        let mut x = x0.clone();
        let mut e = self.problem.residual(&x);
        let mut j = self.problem.jacobian(&x);
        let mut a = j.transpose() * j.clone();
        let mut g = -j.transpose() * e.clone();
        let mut stop = g.norm() < self.eps1;
        let mut u = self.tau * a.diagonal().abs().max();
        let identity = na::DMatrix::identity(x.len(), x.len());
        let mut k = 0;

        while k < self.max_iter && !stop {
            k += 1;
            let rho = 0.0;
            while rho <= 0.0 && !stop {
                let h = a.clone() + u * identity.clone();
                let delta = self.problem.solve(&h, &g);
                if delta.norm() <= self.eps2 * x.norm() {
                    stop = true;
                } else {
                    let x1 = x.clone() + delta.clone();
                    let e1 = self.problem.residual(&x1);
                    let rho = (e.norm().powi(2) - e1.norm().powi(2)) / (delta.transpose() * (u * delta + g.clone()))[0];
                    if rho > 0.0 {
                        x = x1;
                        e = e1;
                        j = self.problem.jacobian(&x);
                        a = j.transpose() * j.clone();
                        g = -j.transpose() * e.clone();
                        stop = g.norm() < self.eps1 || e.norm() < self.eps3;
                        u *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                        v = 2.0;
                    } else {
                        u *= v;
                        v *= 2.0;
                    }   
                }
            }
        }
        println!("f: {}", e);
        println!("F: {}", 0.5 * e.norm().powi(2));

        x
    }

}
