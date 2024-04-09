use nalgebra as na;

use crate::graph::*;

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

    fn is_fixed(&self) -> bool {
        self.fixed
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

    fn is_fixed(&self) -> bool {
        self.fixed
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