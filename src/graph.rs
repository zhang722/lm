use std::{ rc::Rc };
use nalgebra as na;

pub trait Vertex {
    fn params(&self) -> &na::DVector<f64>;
    fn plus(&mut self, delta: &na::DVector<f64>);
}

type VertexBase = Rc<dyn Vertex>;

pub trait Edge {
    fn vertex0(&self) -> VertexBase;
    fn vertex1(&self) -> VertexBase;
    fn residual(&self) -> na::DVector<f64>;
    fn jacobian0(&self) -> na::DMatrix<f64>;
    fn jacobian1(&self) -> na::DMatrix<f64>;
}

type EdgeBase = Rc<dyn Edge>;

pub struct Graph {
    vetices: Vec<Vec<VertexBase>>,
    edges: Vec<EdgeBase>,
}

impl Graph {
    pub fn update_params(&mut self, delta: &na::DVector<f64>) {
        let mut idx = 0;
        for vertex_set in self.vetices.iter_mut() {
            for vertex in vertex_set.iter_mut() {
                let param_len = vertex.as_ref().params().len();
                let mut vertex_delta = na::DVector::<f64>::zeros(param_len);
                vertex_delta.view_mut((0, 0), (param_len, 1)).copy_from(
                    &delta.view_mut((idx, 0), (param_len, 1))
                ); 
                vertex.plus(&vertex_delta);
            }
        } 
    }
}

pub struct CameraVertex {
    params: na::DVector<f64>,
}

impl Vertex for CameraVertex {
    fn params(&self) -> &na::DVector<f64> {
        &self.params
    }

    fn plus(&mut self, delta: &na::DVector<f64>) {
        self.params += delta;
    }
}

pub struct PointVertex {
    params: na::DVector<f64>,
}

impl Vertex for PointVertex {
    fn params(&self) -> &na::DVector<f64> {
        &self.params
    }

    fn plus(&mut self, delta: &na::DVector<f64>) {
        self.params += delta;
    }
}

pub struct Point3dProjectEdge {
    vertex0: Rc<CameraVertex>,
    vertex1: Rc<PointVertex>,
    sigma: na::DMatrix<f64>,
    measurement: na::DVector<f64>,
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

/// Converts an NAlgebra Isometry to a 6-Vector Lie Algebra representation
/// of a rigid body transform.
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn log_map(input: &na::Isometry3<f64>) -> na::Vector6<f64> {
    let t: na::Vector3<f64> = input.translation.vector;

    let quat = input.rotation;
    let theta: f64 = 2.0 * (quat.scalar()).acos();
    let half_theta = 0.5 * theta;
    let mut omega = na::Vector3::<f64>::zeros();

    let mut v_inv = na::Matrix3::<f64>::identity();
    if theta > 1e-6 {
        omega = quat.vector() * theta / (half_theta.sin());
        let ssym_omega = skew_sym(omega);
        v_inv -= ssym_omega * 0.5;
        v_inv += ssym_omega * ssym_omega * (1.0 - half_theta * half_theta.cos() / half_theta.sin())
            / (theta * theta);
    }

    let mut ret = na::Vector6::<f64>::zeros();
    ret.fixed_view_mut::<3, 1>(0, 0).copy_from(&(v_inv * t));
    ret.fixed_view_mut::<3, 1>(3, 0).copy_from(&omega);

    ret
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
    Rsw: &na::Rotation3<f64>,
) -> na::Matrix3<f64>
{
    let mut jac = na::Matrix3::<f64>::zeros();
    jac.copy_from(Rsw.matrix());
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

impl Edge for Point3dProjectEdge {
    fn vertex0(&self) -> VertexBase {
        self.vertex0.clone()
    }

    fn vertex1(&self) -> VertexBase {
        self.vertex1.clone()
    }

    fn residual(&self) -> na::DVector<f64> {
        let camera = &self.vertex0.as_ref().params;
        let point3d = &self.vertex1.as_ref().params;

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

    fn jacobian0(&self) -> na::DMatrix<f64> {
        let camera = &self.vertex0.as_ref().params;
        let point3d = &self.vertex1.as_ref().params;

        let mut jac = na::DMatrix::zeros(2, camera.len());

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
        
        jac.fixed_view_mut::<2, 6>(0, 0)
            .copy_from(&jacobian_r_wrt_pose);
        jac.fixed_view_mut::<2, 3>(0, 6)
            .copy_from(&jacobian_r_wrt_intrinsics);
        
        jac
    }

    fn jacobian1(&self) -> na::DMatrix<f64> {
        let camera = &self.vertex0.as_ref().params;
        let point3d = &self.vertex1.as_ref().params;

        let mut jac = na::DMatrix::zeros(2, point3d.len());

        let rotation = na::Rotation3::new(
            na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
        );
        let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
        let intrinsics = na::Vector3::<f64>::new(camera[6], camera[7], camera[8]);

        let ps = rotation * point3d + t;
        let pn = -ps / ps.z;
        let pn = pn.fixed_view::<2, 1>(0, 0).clone_owned();

        let jacobian_r_wrt_pw = jacobian_pp_wrt_pn(&pn, &intrinsics) 
            * jacobian_pn_wrt_ps(&ps) 
            * jacobian_ps_wrt_pw(&rotation);
        jac.fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&jacobian_r_wrt_pw); 

        jac
    }
}