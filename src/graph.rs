use std::{ collections::HashMap, cell::RefCell, rc::Rc};

use nalgebra as na;

pub trait Id {
    fn id(&self) -> usize;
    fn id_mut(&mut self) -> &mut usize;
}

pub trait Vertex : Id {
    fn edges(&self) -> &Vec<usize>;
    fn edges_mut(&mut self) -> &mut Vec<usize>;
    fn params(&self) -> &na::DVector<f64>;
    fn plus(&mut self, delta: &na::DVector<f64>);

    fn hessian_index(&self) -> usize;
    fn hessian_index_mut(&mut self) -> &mut usize;

    fn is_fixed(&self) -> bool {
        false
    }

    fn dimension(&self) -> usize {
        self.params().len()
    }
    fn add_edge(&mut self, id: usize) {
        self.edges_mut().push(id);
    }
}

pub type VertexBase = Rc<RefCell<dyn Vertex>>;

pub trait Edge: Id {
    fn vertex(&self, ith: usize) -> VertexBase;
    fn vertices(&self) -> &Vec<VertexBase>;
    fn vertices_mut(&mut self) -> &mut Vec<VertexBase>;
    fn residual(&self) -> na::DVector<f64>;
    fn jacobian(&self, ith: usize) -> na::DMatrix<f64>;
    fn sigma(&self) -> na::DMatrix<f64>;

    fn dimension(&self) -> usize {
        self.residual().len()
    }
    fn add_vertex(&mut self, vertex: VertexBase) {
        self.vertices_mut().push(vertex.clone());
        vertex.borrow_mut().add_edge(self.id());
    }
}

pub type EdgeBase = Rc<RefCell<dyn Edge>>;

pub struct LmParams {
    tau: f64,
    eps1: f64,
    eps2: f64,
    eps3: f64,
    v: f64,
    max_iter: usize,
    u: f64,
}

impl LmParams {
    pub fn new(tau: f64, eps1: f64, eps2: f64, eps3: f64, v: f64, max_iter: usize) -> Self {
        Self { tau, eps1, eps2, eps3, v, max_iter, u: 0.0 }
    }
}

impl Default for LmParams {
    fn default() -> Self {
        Self::new(1e-3, 1e-15, 1e-6, 1e-15, 2.0, 100)
    }
}

pub struct Graph {
    pub vertices: Vec<Vec<VertexBase>>,
    pub edges: HashMap<usize, EdgeBase>,
    pub lm_params: LmParams,
    hessian: na::DMatrix<f64>,
    b: na::DVector<f64>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: HashMap::<usize, EdgeBase>::new(),
            lm_params: LmParams::default(),
            hessian: na::DMatrix::<f64>::zeros(0, 0),
            b: na::DVector::<f64>::zeros(0),
        }
    }

    pub fn print(&self) {
        println!("{} vertices:", self.vertices.len());
        for vertex_set in &self.vertices {
            for vertex in vertex_set {
                print!("id: {}, ", vertex.borrow().id());
                print!("edges: {:?}, ", vertex.borrow().edges());
                print!("params: {}", vertex.borrow().params().transpose());
            }
        }

        println!("{} edges:", self.edges.len());
        for edge in self.edges.values() {
            print!("id: {}, ", edge.borrow().id());
            print!("vertices: {:?}", edge.borrow().vertices().iter().map(|x| { x.borrow().id() }).collect::<Vec<_>>());
            println!("residual: {}", edge.borrow().residual().transpose());
            // println!("jacobian: {}", edge.borrow().jacobian(0));
        }
    }

    pub fn add_vertex(&mut self, vertex: &VertexBase, ith_set: usize) {
        self.vertices[ith_set].push(vertex.clone());
    }

    pub fn add_vertex_set(&mut self, vertex_set: Vec<VertexBase>) {
        self.vertices.push(vertex_set);
    }

    pub fn add_edge(&mut self, edge: &EdgeBase) {
        self.edges.insert(edge.borrow().id(), edge.clone());
    }

    pub fn vertex(&self, id: usize) -> Option<VertexBase> {
        for vertex_set in &self.vertices {
            for vertex in vertex_set {
                if vertex.borrow().id() == id {
                    return Some(vertex.clone())
                }
            }
        }

        None
    }

    pub fn vertex_set(&self, ith: usize) -> &[VertexBase] {
        self.vertices[ith].as_slice()
    }

    pub fn edge(&self, id: usize) -> Option<EdgeBase> {
        self.edges.get(&id).cloned()
    }

    pub fn params_dim(&self) -> usize {
        let mut dim = 0;
        for vertex_set in self.vertices.iter() {
            for vertex in vertex_set {
                if vertex.borrow().is_fixed() {
                    continue;
                }
                dim += vertex.borrow().dimension();
            }
        }

        dim
    }

    pub fn update_params(&mut self, delta: &na::DVector<f64>) {
        let mut idx = 0;
        for vertex_set in self.vertices.iter() {
            for vertex in vertex_set {
                if vertex.borrow().is_fixed() {
                    continue;
                }
                let dim: usize = vertex.borrow().dimension();
                let delta_vertex = delta.rows(idx, dim).clone_owned();
                vertex.borrow_mut().plus(&delta_vertex);

                idx += dim;
            }
        }
    }

    pub fn vertex2param(&self) -> na::DVector<f64> {
       let mut dim_total = 0;
        for vertex_set in self.vertices.iter() {
            for vertex in vertex_set {
                dim_total += vertex.borrow().dimension();
            }
        }

        let mut idx = 0;
        let mut params = na::DVector::<f64>::zeros(dim_total);
        for vertex_set in self.vertices.iter() {
            for vertex in vertex_set {
                let dim = vertex.borrow().dimension();
                params.rows_mut(idx, dim).copy_from(
                    vertex.borrow().params()
                );
                idx += dim;
            }
        }

        params
    }

    pub fn params_norm(&self) -> f64 {
        self.vertex2param().norm() 
    }

    fn init_u(&self) -> f64 {
        0.01
    }

    pub fn prepare_hessian_index(&self) {
        let mut idx = 0;
        for vertex_set in self.vertices.iter() {
            for vertex in vertex_set {
                *vertex.borrow_mut().hessian_index_mut() = idx;
                idx += vertex.borrow().dimension();
            }
        }
    }

    pub fn calculate_residual(&self) -> f64 {
        let mut chi2 = 0.0;
        for edge in self.edges.values() {
            let residual = edge.borrow().residual();
            chi2 += (residual.transpose() * edge.borrow().sigma() * residual)[(0, 0)];
        }

        chi2
    }

    pub fn calculate_jt_residual(&self) -> na::DVector<f64> {
        let mut jt_residual = na::DVector::<f64>::zeros(self.params_dim());
        for edge in self.edges.values() {
            for (idx_v, v) in edge.borrow().vertices().iter().enumerate() {
                if v.borrow().is_fixed() {
                    continue;
                }

                let idx_hessian = v.borrow().hessian_index();
                let dim = v.borrow().dimension();
                let jacobian = edge.borrow().jacobian(idx_v);
                let residual = edge.borrow().residual();
                let mut temp = jt_residual.rows(idx_hessian, dim).clone_owned();
                temp -= &(jacobian.transpose() * edge.borrow().sigma() * residual);
                jt_residual.rows_mut(idx_hessian, dim).copy_from(&temp);
            }
        }

        jt_residual
    }

    pub fn make_normal_equation(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut a = na::DMatrix::<f64>::zeros(self.params_dim(), self.params_dim());
        let mut b = na::DVector::<f64>::zeros(self.params_dim());
        for edge in self.edges.values() {
            let v_size = edge.borrow().vertices().len();
            for idx_v_i in 0..v_size {
                let v_i = edge.borrow().vertex(idx_v_i);
                if v_i.borrow().is_fixed() {
                    continue;
                }
                let idx_hessian_i = v_i.borrow().hessian_index();
                let dim_i = v_i.borrow().dimension();
                for idx_v_j in idx_v_i..v_size {
                    let v_j = edge.borrow().vertex(idx_v_j);
                    if v_j.borrow().is_fixed() {
                        continue;
                    }
                    let idx_hessian_j = v_j.borrow().hessian_index();
                    let dim_j = v_j.borrow().dimension();
                    let mut hessian = a.view((idx_hessian_i, idx_hessian_j), (dim_i, dim_j)).clone_owned();
                    let temp = edge.borrow().jacobian(idx_v_i).transpose() * edge.borrow().sigma() * edge.borrow().jacobian(idx_v_j);
                    hessian += &temp;
                    a.view_mut((idx_hessian_i, idx_hessian_j), (dim_i, dim_j)).copy_from(&hessian);

                    if idx_v_i != idx_v_j {
                        let mut hessian = a.view((idx_hessian_j, idx_hessian_i), (dim_j, dim_i)).clone_owned();
                        let temp = edge.borrow().jacobian(idx_v_i).transpose() * edge.borrow().sigma() * edge.borrow().jacobian(idx_v_j);
                        hessian += &temp.transpose();
                        a.view_mut((idx_hessian_j, idx_hessian_i), (dim_j, dim_i)).copy_from(&hessian);
                    }
                }

                let mut temp = b.rows(idx_hessian_i, dim_i).clone_owned();
                temp -= &(edge.borrow().jacobian(idx_v_i).transpose() * edge.borrow().sigma() * edge.borrow().residual());
                b.rows_mut(idx_hessian_i, dim_i).copy_from(&temp);
            }
        }

        self.hessian = a + &(na::DMatrix::<f64>::identity(self.params_dim(), self.params_dim()) * self.lm_params.tau);
        self.b = b;

        Ok(())
    }

    fn calculate_v_inv(&self) -> Option<na::DMatrix<f64>> {
        assert!(self.vertices.len() > 1);
        let mut dim_point = 0;
        for vertex in self.vertices[1].iter() {
            if vertex.borrow().is_fixed() {
                continue;
            }
            let dim = vertex.borrow().dimension();
            dim_point += dim;
        }
        let mut hmm_inv = na::DMatrix::<f64>::zeros(dim_point, dim_point);
        for vertex in self.vertices[1].iter() {
            if vertex.borrow().is_fixed() {
                continue;
            }
            let idx_hessian = vertex.borrow().hessian_index();
            let dim = vertex.borrow().dimension();
            let v_inv = self.hessian.view((idx_hessian, idx_hessian), (dim, dim))
                .try_inverse()?;
            let idx_v_inv = idx_hessian + dim_point - self.hessian.nrows();
            hmm_inv.view_mut((idx_v_inv, idx_v_inv), (dim, dim)).copy_from(&v_inv);
        }

        Some(hmm_inv)
    } 

    fn calculate_delta_point(&self, v_inv: &na::DMatrix<f64>, rhs: &na::DVector<f64>) -> Option<na::DVector<f64>> {
        assert!(self.vertices.len() > 1);
        let mut delta_point = na::DVector::<f64>::zeros(v_inv.nrows());
        let mut idx = 0;
        for vertex in self.vertices[1].iter() {
            if vertex.borrow().is_fixed() {
                continue;
            }
            let dim = vertex.borrow().dimension();
            delta_point.rows_mut(idx, dim).copy_from(
                &(v_inv.view((idx, idx), (dim, dim)) * rhs.rows(idx, dim))
            );
            idx += dim;
        }

        Some(delta_point)
    }

    pub fn calculate_delta_step(&mut self) -> Option<na::DVector<f64>> {
        self.make_normal_equation().ok()?;
        let v_inv = self.calculate_v_inv()?;
        let dim_point = v_inv.nrows();
        let dim_pose = self.hessian.nrows() - dim_point;
        let hpp = self.hessian.view((0, 0), (dim_pose, dim_pose));
        let hpm = self.hessian.view((0, dim_pose), (dim_pose, dim_point));
        let hmp = self.hessian.view((dim_pose, 0), (dim_point, dim_pose));
        let bp = self.b.rows(0, dim_pose);
        let bm = self.b.rows(dim_pose, dim_point);

        let delta_pose = (hpp - hpm * &v_inv * hmp).lu().solve(&(bp - hpm * &v_inv * bm))?;
        let delta_point = self.calculate_delta_point(&v_inv, &(bm - hmp * &delta_pose))?;

        let mut delta = na::DVector::<f64>::zeros(self.hessian.nrows());
        delta.rows_mut(0, dim_pose).copy_from(&delta_pose);
        delta.rows_mut(dim_pose, dim_point).copy_from(&delta_point);

        Some(delta)
    }

    pub fn optimize(&mut self) -> Option<()>{
        let mut v = self.lm_params.v;
        let mut e = self.calculate_residual();
        let mut jt_residual = self.calculate_jt_residual();
        let mut stop = jt_residual.abs().max() < self.lm_params.eps1;
        let mut k = 0;
        self.prepare_hessian_index();

        while k < self.lm_params.max_iter && !stop {
            println!("k: {}\t e: {}", k, e);
            k += 1;
            let mut rho = 0.0;
            if k == 1 {
                self.init_u();
            }
            while rho <= 0.0 && !stop {
                let delta = self.calculate_delta_step()?;
                if delta.norm() <= self.lm_params.eps2 * self.params_norm() {
                    stop = true;
                } else {
                    self.update_params(&delta);
                    let e1 = self.calculate_residual();
                    rho = (e - e1) / ((delta.transpose() * (self.lm_params.u * delta + &jt_residual))[0] + 1e-3);
                    if rho > 0.0 {
                        e = e1;
                        jt_residual = self.calculate_jt_residual();
                        stop = jt_residual.abs().max() < self.lm_params.eps1 || e < self.lm_params.eps3;
                        self.lm_params.u *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                        v = 2.0;
                    } else {
                        self.lm_params.u *= v;
                        v *= 2.0;
                    }   
                }
            }
        }

        Some(())
    }
}

