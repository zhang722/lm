use nalgebra as na;

use crate::graph::*;

type Vector8<T> = na::Matrix<T, na::U8, na::U1, na::ArrayStorage<T, 8, 1>>;
/*fx, fy, cx, cy, k1, k2, p1, p2*/
pub type CameraInstrinsics = Vector8<f64>;

/// Projects a point in camera coordinates into the image plane
/// producing a floating-point pixel value
pub fn ps2pdn(
    intrinsics: &CameraInstrinsics, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    ps: &na::Vector3<f64>,
) -> na::Vector2<f64> {
    let k1 = intrinsics[4];
    let k2 = intrinsics[5];
    let p1 = intrinsics[6];
    let p2 = intrinsics[7];

    let xn = ps.x / ps.z;
    let yn = ps.y / ps.z;
    let rn2 = xn * xn + yn * yn;
    na::Vector2::<f64>::new(
        xn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p1 * xn * yn + p2 * (rn2 + 2.0 * xn * xn),
        yn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p2 * xn * yn + p1 * (rn2 + 2.0 * yn * yn)
    )
}

pub fn pdn2pp(
    intrinsics: &CameraInstrinsics, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    pdn: &na::Vector2<f64>,
) -> na::Vector2<f64> {
    let fx = intrinsics[0];
    let fy = intrinsics[1];
    let cx = intrinsics[2];
    let cy = intrinsics[3];
    na::Vector2::<f64>::new(
        fx * pdn[0] + cx,
        fy * pdn[1] + cy
    )
}

fn jacobian_pp_wrt_pn(
    pn: &na::Vector2<f64>, 
    intrinsics: &CameraInstrinsics,
) -> na::Matrix2<f64> 
{
    let fx = intrinsics[0];
    let fy = intrinsics[1];
    let k1 = intrinsics[4];
    let k2 = intrinsics[5];
    let p1 = intrinsics[6];
    let p2 = intrinsics[7];
    
    let x = pn[0];
    let y = pn[1];

    let rn2 = x.powi(2) + y.powi(2);
    let rn4 = rn2.powi(2);

    na::Matrix2::<f64>::new(
        fx * (k1 * rn2 + k2 * rn4 + 2.0 * p1 * y + 4.0 * p2 * x + 1.0), 2.0 * fx * p1 * x,
        2.0 * fy * p2 * y, fy * (k1 * rn2 + k2 * rn4 + 4.0 * p1 * y + 2.0 * p2 * x + 1.0)
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

    na::Matrix2x3::<f64>::new(
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
    jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&na::Matrix3::<f64>::identity());
    jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&-skew_sym(na::Vector3::<f64>::new(x, y, z)));
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

fn decode_camera_pose(
    pose: &na::DVector<f64>,
) -> (na::Vector3<f64>, na::Rotation3<f64>) {
    let translation = na::Vector3::<f64>::new(pose[0], pose[1], pose[2]);
    let rotation = na::Vector3::<f64>::new(pose[3], pose[4], pose[5]);
    (translation, na::Rotation3::<f64>::new(rotation))
}

pub struct Point3dProjectWithIntrinsicEdge {
    pub id: usize,
    pub vertices: Vec<VertexBase>,
    pub sigma: na::DMatrix<f64>,
    pub measurement: na::DVector<f64>,        
    pub intrinsic: CameraInstrinsics,
}

impl Id for Point3dProjectWithIntrinsicEdge {
    fn id(&self) -> usize {
        self.id
    }

    fn id_mut(&mut self) -> &mut usize {
        &mut self.id
    }
}

impl Edge for Point3dProjectWithIntrinsicEdge {
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

        let (t, rotation) = decode_camera_pose(camera);

        let intrinsics = self.intrinsic;

        let ps = rotation * point3d + t;
        let pn = ps2pdn(&intrinsics, &ps);
        let pp = pdn2pp(&intrinsics, &pn);
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

        let (t, rotation) = decode_camera_pose(camera);
        let intrinsics = self.intrinsic;

        let ps = rotation * point3d + t;
        let pn = ps2pdn(&intrinsics, &ps);

        let jacobian_r_wrt_pose = jacobian_pp_wrt_pn(&pn, &intrinsics) 
            * jacobian_pn_wrt_ps(&ps) 
            * jacobian_ps_wrt_pose(&ps);

        let jacobian_r_wrt_pw = jacobian_pp_wrt_pn(&pn, &intrinsics) 
            * jacobian_pn_wrt_ps(&ps) 
            * jacobian_ps_wrt_pw(&rotation);
        let mut jacobian_r_wrt_camera = na::DMatrix::<f64>::zeros(
            jacobian_r_wrt_pose.nrows(), jacobian_r_wrt_pose.ncols());
        jacobian_r_wrt_camera.view_mut((0, 0), jacobian_r_wrt_pose.shape()).copy_from(&jacobian_r_wrt_pose);
        
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
