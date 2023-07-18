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


type CameraInstrinsics = na::Vector3<f64>;

fn jacobian_pp_wrt_pn(
    pn: &na::Vector2<f64>, 
    intrinsics: &CameraInstrinsics,
) -> na::Matrix2<f64> 
{
    let f = intrinsics[0];
    let k1 = intrinsics[1];
    let k2 = intrinsics[2];
    let x = pn[0];
    let y = pn[1];

    let rn2 = x.powi(2) + y.powi(2);
    let rn4 = rn2.powi(2);

    na::Matrix2::<f64>::new(
        f * (k1 * rn2 + k2 * rn4 + 1.0), 0.0,
        0.0, f * (k1 * rn2 + k2 * rn4 + 1.0)
    )
}

fn jacobian_pn_wrt_ps(
    ps: &na::Vector3<f64>,
) -> na::Matrix2x3<f64>
{
    let x = ps[0];
    let y = ps[1];
    let z = ps[2];
    let z2 = z.powi(2);

    -na::Matrix2x3::<f64>::new(
        1.0 / z, 0.0, -x / z2, 
        0.0, 1.0 / z, -y / z2)
}

/// Produces a skew-symmetric or "cross-product matrix" from
/// a 3-vector. This is needed for the `exp_map` and `log_map`
/// functions
fn skew_sym(v: na::Vector3<f64>) -> na::Matrix3<f64> {
    let mut ss = na::Matrix3::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

fn jacobian_ps_wrt_pose(
    ps: &na::Vector3<f64>,
) -> na::Matrix3x6<f64>
{
    let x = ps[0];
    let y = ps[1];
    let z = ps[2];

    let mut jac = na::Matrix3x6::<f64>::zeros();
    jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&na::Matrix3::<f64>::identity());
    jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&-skew_sym(na::Vector3::<f64>::new(x, y, z)));
    jac
}

fn jacobian_ps_wrt_pw(
    r_sw: &na::Rotation3<f64>,
) -> na::Matrix3<f64>
{
    let mut jac = na::Matrix3::<f64>::zeros();
    jac.copy_from(r_sw.matrix());
    jac
}

fn jacobian_pp_wrt_intrinsics(
    pn: &na::Vector2<f64>,
    intrinsics: &CameraInstrinsics,
) -> na::Matrix2x3<f64>
{
    let f = intrinsics[0];
    let k1 = intrinsics[1];
    let k2 = intrinsics[2];
    let x = pn[0];
    let y = pn[1];

    let rn2 = x.powi(2) + y.powi(2);
    let rn4 = rn2.powi(2);

    na::Matrix2x3::<f64>::new(
        x * (k1 * rn2 + k2 * rn4 + 1.0), f * rn2 * x, f * rn4 * x,
        y * (k1 * rn2 + k2 * rn4 + 1.0), f * rn2 * y, f * rn4 * y
    )
}

pub struct CameraVertex {
    pub id: usize,
    pub params: na::DVector<f64>,
    pub edges: Vec<usize>,
    pub fixed: bool,
    pub hessian_index: usize,
}

impl Id for CameraVertex {
    fn id(&self) -> usize {
        self.id
    }

    fn id_mut(&mut self) -> &mut usize {
        &mut self.id
    }
}

impl Vertex for CameraVertex {
    fn edges(&self) -> &Vec<usize> {
        &self.edges
    }

    fn edges_mut(&mut self) -> &mut Vec<usize> {
        &mut self.edges
    }

    fn params(&self) -> &na::DVector<f64> {
        &self.params
    }

    fn plus(&mut self, delta: &na::DVector<f64>) {
        self.params += delta;
    }

    fn hessian_index(&self) -> usize{
        self.hessian_index
    }

    fn hessian_index_mut(&mut self) -> &mut usize {
        &mut self.hessian_index
    }

    fn dimension(&self) -> usize {
        self.params().len()
    }

    fn add_edge(&mut self, id: usize) {
        self.edges_mut().push(id);
    }
}

pub struct PointVertex {
    pub id: usize,
    pub params: na::DVector<f64>,
    pub edges: Vec<usize>,
    pub fixed: bool,
    pub hessian_index: usize,
}

impl Id for PointVertex {
    fn id(&self) -> usize {
        self.id
    }

    fn id_mut(&mut self) -> &mut usize {
        &mut self.id
    }
}

impl Vertex for PointVertex {
    fn edges(&self) -> &Vec<usize> {
        &self.edges
    }

    fn edges_mut(&mut self) -> &mut Vec<usize> {
        &mut self.edges
    }

    fn params(&self) -> &na::DVector<f64> {
        &self.params
    }

    fn plus(&mut self, delta: &na::DVector<f64>) {
        self.params += delta;
    }

    fn hessian_index(&self) -> usize {
        self.hessian_index
    }
    
    fn hessian_index_mut(&mut self) -> &mut usize {
        &mut self.hessian_index
    }
}

pub struct Point3dProjectEdge {
    pub id: usize,
    pub vertices: Vec<VertexBase>,
    pub sigma: na::DMatrix<f64>,
    pub measurement: na::DVector<f64>,        
}

impl Id for Point3dProjectEdge {
    fn id(&self) -> usize {
        self.id
    }

    fn id_mut(&mut self) -> &mut usize {
        &mut self.id
    }
}

impl Edge for Point3dProjectEdge {
    fn vertex(&self, ith: usize) -> VertexBase {
        self.vertices[ith].clone()
    }

    fn vertices(&self) -> &Vec<VertexBase> {
        &self.vertices
    }

    fn vertices_mut(&mut self) -> &mut Vec<VertexBase> {
        &mut self.vertices
    }

    fn residual(&self) -> na::DVector<f64> {
        let camera = self.vertex(0);
        let point3d = self.vertices[1].borrow();
        let camera = camera.borrow();
        let camera = camera.params();
        let point3d = point3d.params();

        let rotation = na::Rotation3::new(
            na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
        );
        let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
        let f = camera[6];
        let k1 = camera[7];
        let k2 = camera[8];

        let ps = rotation * point3d + t;
        let pn = -ps / ps.z;
        let pn = pn.fixed_view::<2, 1>(0, 0);
        let pp = f * (1.0 + k1 * pn.norm().powi(2) + k2 * pn.norm().powi(4)) * pn;
        let mut res = na::DVector::zeros(pp.len());
        res.fixed_view_mut::<2, 1>(0, 0).copy_from(
            &(pp - na::Vector2::new(self.measurement[0], self.measurement[1]))
        );

        res
    }

    fn jacobian(&self, ith: usize) -> na::DMatrix<f64> {
        let camera = self.vertices[0].borrow();
        let camera = camera.params();
        let point3d = self.vertices[1].borrow();
        let point3d = point3d.params();

        let rotation = na::Rotation3::new(
            na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
        );
        let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
        let intrinsics = na::Vector3::<f64>::new(camera[6], camera[7], camera[8]);

        let ps = rotation * point3d + t;
        let pn = -ps / ps.z;
        let pn = pn.fixed_view::<2, 1>(0, 0).clone_owned();

        let jacobian_r_wrt_pose = jacobian_pp_wrt_pn(&pn, &intrinsics) 
            * jacobian_pn_wrt_ps(&ps) 
            * jacobian_ps_wrt_pose(&ps);
        let jacobian_r_wrt_intrinsics = jacobian_pp_wrt_intrinsics(&pn, &intrinsics);

        let jacobian_r_wrt_pw = jacobian_pp_wrt_pn(&pn, &intrinsics) 
            * jacobian_pn_wrt_ps(&ps) 
            * jacobian_ps_wrt_pw(&rotation);
        let mut jacobian_r_wrt_camera = na::DMatrix::<f64>::zeros(
            jacobian_r_wrt_pose.nrows(), jacobian_r_wrt_pose.ncols() + jacobian_r_wrt_intrinsics.ncols());
        jacobian_r_wrt_camera.view_mut((0, 0), jacobian_r_wrt_pose.shape()).copy_from(&jacobian_r_wrt_pose);
        jacobian_r_wrt_camera.view_mut((0, jacobian_r_wrt_pose.ncols()), jacobian_r_wrt_intrinsics.shape()).copy_from(&jacobian_r_wrt_intrinsics);
        
        if ith == 0 {
            jacobian_r_wrt_camera
        } else {
            jacobian_r_wrt_pw.view((0, 0), jacobian_r_wrt_pw.shape()).clone_owned()
        }
    }

    fn sigma(&self) -> na::DMatrix<f64> {
        na::DMatrix::<f64>::identity(self.measurement.len(), self.measurement.len())
    }
}


#[cfg(test)]
mod tests{
    use std::{rc::Rc, cell::RefCell};

    use nalgebra as na;


    #[test]
    fn test_graph() {
        let camera_vertices: Vec<super::VertexBase> = (0..3usize).into_iter().map(|x| {
            Rc::new(RefCell::new(super::CameraVertex {
                id: x,
                params: na::DVector::<f64>::zeros(9),
                edges: Vec::new(),
                fixed: false,
                hessian_index: 0,
            })) as super::VertexBase
        }).collect::<Vec<super::VertexBase>>();

        let point_vertices = (0..10usize).into_iter().map(|x| {
            Rc::new(RefCell::new(super::PointVertex {
                id: x + 3,
                params: na::DVector::<f64>::zeros(3),
                edges: Vec::new(),
                fixed: false,
                hessian_index: 0,
            })) 
            as super::VertexBase
        }).collect::<Vec<_>>();

        let edge1: Rc<RefCell<dyn super::Edge>> = Rc::new(RefCell::new(super::Point3dProjectEdge {
            id: 1,
            vertices: Vec::new(),
            sigma: na::DMatrix::<f64>::zeros(2, 2),
            measurement: na::DVector::<f64>::zeros(2),
        }));
        let mut graph = super::Graph::default();
        graph.add_vertex_set(camera_vertices);
        graph.add_vertex_set(point_vertices);
        edge1.borrow_mut().add_vertex(graph.vertices[0][0].clone());
        edge1.borrow_mut().add_vertex(graph.vertices[1][0].clone());
        graph.add_edge(&edge1);
        for edge in graph.vertices[0][0].borrow().edges() {
            println!("edge: {}", edge);
        }
        for vertex in edge1.borrow().vertices().iter() {
            println!("vertex: {}", vertex.borrow().id());
        }

        graph.optimize();
        for vertex_set in graph.vertices {
            for vertex in vertex_set {
                println!("vertex: {}", vertex.borrow().params());
            }
        }
    }
}