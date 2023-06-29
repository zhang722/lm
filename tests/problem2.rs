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
        let a = na::DVector::<f64>::from_vec(
            (1..=M).into_iter().map(|i| i as f64).collect::<Vec<_>>()
        ) * na::RowDVector::<f64>::from_vec(
            (1..=N).into_iter().map(|i| i as f64).collect::<Vec<_>>()
        );

        a * x - ev
    }

    fn jacobian(&self, x: &na::DVector<f64>) -> na::DMatrix<f64> {
        let mut ev = na::DVector::<f64>::zeros(M);
        ev.fill(1.0);

        na::DVector::<f64>::from_vec(
            (1..=M).into_iter().map(|i| i as f64).collect::<Vec<_>>()
        ) * na::RowDVector::<f64>::from_vec(
            (1..=N).into_iter().map(|i| i as f64).collect::<Vec<_>>()
        )
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
    let k = (na::RowDVector::<f64>::from_vec(
            (1..=N).into_iter().map(|i| i as f64).collect::<Vec<_>>()
    ) * x)[0];
    println!("k = {}", k);
}