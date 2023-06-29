use nalgebra as na;

struct TestProblem;

const M: usize = 6;
const N: usize = 4;

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
        let mut ev = na::DVector::<f64>::zeros(M);
        ev.fill(1.0);
        let mut a = na::DMatrix::<f64>::zeros(M, N);
        a.fixed_view_mut::<N, N>(0, 0).copy_from(&(
            na::Matrix4::<f64>::identity() - 2.0 / (M as f64) * ones::<N, N>()
        ));
        a.fixed_view_mut::<{M-N}, N>(N, 0).copy_from(&(
            - 2.0 / (M as f64) * ones::<{M-N}, N>()
        ));

        a * x - ev
    }

    fn jacobian(&self, x: &na::DVector<f64>) -> na::DMatrix<f64> {
        let mut ev = na::DVector::<f64>::zeros(M);
        ev.fill(1.0);
        let mut a = na::DMatrix::<f64>::zeros(M, N);
        a.fixed_view_mut::<N, N>(0, 0).copy_from(&(
            na::Matrix4::<f64>::identity() - 2.0 / (M as f64) * ones::<N, N>()
        ));
        a.fixed_view_mut::<{M-N}, N>(N, 0).copy_from(&(
            - 2.0 / (M as f64) * ones::<{M-N}, N>()
        ));

        a
    }
}

#[test]
fn test_problem1() {
    let problem = TestProblem;
    let mut x0 = na::DVector::<f64>::zeros(N); 
    x0.fill(1.0);
    let solver = lm::LM::default(Box::new(problem));
    let x = solver.optimize(&x0);
    println!("x = {}", x);
}