use nalgebra as na;

struct TestProblem;

const M: usize = 2;
const N: usize = 2;

fn ones<const M: usize, const N: usize>() -> na::OMatrix::<f64, na::Const<M>, na::Const<N>> {
    let mut m = na::OMatrix::<f64, na::Const<M>, na::Const<N>>::zeros();
    m.fill(1.0);
    m
}

impl lm::LMProblem for TestProblem {
    fn solve(&self, x: &na::DMatrix<f64>, y: &na::DVector<f64>) -> na::DVector<f64> {
        x.clone().lu().solve(y).unwrap()
    }

    fn residual(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        na::DVector::<f64>::from_vec(vec![
            10.0 * (x[1] - x[0].powi(2)),
            1.0 - x[0]
        ])
    }

    fn jacobian(&self, x: &na::DVector<f64>) -> na::DMatrix<f64> {
        na::DMatrix::<f64>::from_row_slice(M, N, &[
            -20.0 * x[0], 10.0, 
            -1.0, 0.0
        ])
    }
}

#[test]
fn test_problem1() {
    let problem = TestProblem;
    let x0 = na::DVector::<f64>::from_vec(vec![-1.2, 1.0]); 
    let solver = lm::LM::default(Box::new(problem));
    let x = solver.optimize(&x0);
    println!("x = {}", x);
}