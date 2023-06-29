use std::{fmt, ops::Range, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

use nalgebra::{
    DMatrix, Matrix2x3, Matrix2x6, Matrix3, Matrix3x4, Matrix6, Matrix6x1, Matrix6x3, MatrixXx1,
    MatrixXx4, Vector2, Vector3, Vector4,
};

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;


const PERSPECTIVE_VALUE_RANGE: f64 = 100.0;
const BUNDLE_ADJUSTMENT_MAX_ITERATIONS: usize = 1000;
const OUTLIER_FILTER_STDEV_THRESHOLD: f64 = 1.0;
const OUTLIER_FILTER_SEARCH_AREA: usize = 5;
const OUTLIER_FILTER_MIN_NEIGHBORS: usize = 10;
const PERSPECTIVE_DISTORTION_SAFETY_RADIUS: f64 = 0.5;
const RANSAC_N: usize = 3;
const RANSAC_K: usize = 10_000_000;
// TODO: this should pe proportional to image size
const RANSAC_INLIERS_T: f64 = 1.0;
const RANSAC_T: f64 = 3.0;
const RANSAC_D: usize = 100;
const RANSAC_D_EARLY_EXIT: usize = 10_000;
const RANSAC_CHECK_INTERVAL: usize = 10_000;

struct BundleAdjustment<'a> {
    cameras: Vec<Camera>,
    projections: Vec<Matrix3x4<f64>>,
    tracks: &'a [Track],
    image_shapes: Vec<(usize, usize)>,
    points3d: &'a mut Vec<Option<Vector3<f64>>>,
    covariance: f64,
    mu: f64,
}

impl BundleAdjustment<'_> {
    const CAMERA_PARAMETERS: usize = 6;
    const INITIAL_MU: f64 = 1E-2;
    const JACOBIAN_H: f64 = 0.001;
    const GRADIENT_EPSILON: f64 = 1E-12;
    const DELTA_EPSILON: f64 = 1E-12;
    const RESIDUAL_EPSILON: f64 = 1E-12;
    const RESIDUAL_REDUCTION_EPSILON: f64 = 0.0;

    fn new<'a>(
        cameras: Vec<Camera>,
        tracks: &'a [Track],
        image_shapes: Vec<(usize, usize)>,
        points3d: &'a mut Vec<Option<Vector3<f64>>>,
    ) -> BundleAdjustment<'a> {
        // For now, identity covariance is acceptable.
        let covariance = 1.0;
        let projections = cameras.iter().map(|camera| camera.projection()).collect();
        BundleAdjustment {
            cameras,
            projections,
            tracks,
            image_shapes,
            points3d,
            covariance,
            mu: BundleAdjustment::INITIAL_MU,
        }
    }

    fn jacobian_view(&self, camera: &Camera, point: &Vector4<f64>, param: usize) -> Vector2<f64> {
        let delta_r = match param {
            0 => Vector3::new(BundleAdjustment::JACOBIAN_H, 0.0, 0.0),
            1 => Vector3::new(0.0, BundleAdjustment::JACOBIAN_H, 0.0),
            2 => Vector3::new(0.0, 0.0, BundleAdjustment::JACOBIAN_H),
            _ => Vector3::zeros(),
        };
        let delta_t = match param {
            3 => Vector3::new(BundleAdjustment::JACOBIAN_H, 0.0, 0.0),
            4 => Vector3::new(0.0, BundleAdjustment::JACOBIAN_H, 0.0),
            5 => Vector3::new(0.0, 0.0, BundleAdjustment::JACOBIAN_H),
            _ => Vector3::zeros(),
        };
        let mut p_plus = camera.clone();
        p_plus.r += delta_r;
        p_plus.t += delta_t;
        let mut p_minus = camera.clone();
        p_minus.r -= delta_r;
        p_minus.t -= delta_t;

        let p_plus = p_plus.projection();
        let p_minus = p_minus.projection();

        let projection_plus = p_plus * point;
        let projection_plus = projection_plus.remove_row(2).unscale(projection_plus.z);
        let projection_minus = p_minus * point;
        let projection_minus = projection_minus.remove_row(2).unscale(projection_minus.z);
        (projection_plus - projection_minus).unscale(2.0 * BundleAdjustment::JACOBIAN_H)
    }

    fn jacobian_a(&self, point3d: &Vector3<f64>, view_j: usize) -> Matrix2x6<f64> {
        // jac is a 2x6 Jacobian for point i and projection matrix parameter j.
        let mut jac = Matrix2x6::zeros();

        let camera = &self.cameras[view_j];
        let point4d = point3d.insert_row(3, 1.0);
        // Calculate Jacobian using finite differences (central difference)
        for i in 0..6 {
            jac.column_mut(i)
                .copy_from(&self.jacobian_view(camera, &point4d, i));
        }

        jac
    }

    fn jacobian_b(&self, point3d: &Vector3<f64>, view_j: usize) -> Matrix2x3<f64> {
        // Point coordinates matrix is converted to a single row with coordinates x, y and z.
        // jac is a 2x3 Jacobian for point i and projection matrix j.
        let mut jac = Matrix2x3::zeros();
        let point4d = point3d.insert_row(3, 1.0);

        let projection = self.projections[view_j];

        // Using a symbolic formula (not finite differences/central difference), check the Rust LM library for more info.
        // P is the projection matrix for image
        // Prc is the r-th row, c-th column of P
        // X is a 3D coordinate (4-component vector [x y z 1])

        // Image contains xpro and ypro (projected point coordinates), affected by projection matrix and the point coordinates.
        // xpro = (P11*x+P12*y+P13*z+P14)/(P31*x+P32*y+P33*z+P34)
        // ypro = (P21*x+P22*y+P23*z+P24)/(P31*x+P32*y+P33*z+P34)
        // To keep things sane, create some aliases
        // Poi -> i-th element of 3D point e.g. x, y, z, or w
        // Pr1 = P11*x+P12*y+P13*z+P14
        // Pr2 = P21*x+P22*y+P23*z+P24
        // Pr3 = P31*x+P32*y+P33*z+P34
        let p_r = projection * point4d;
        // dxpro/dx = (P11*(P32*y+P33*z+P34)-P31*(P12*y+P13*z+P14))/(Pr3^2) = (P11*Pr3[x=0]-P31*Pr1[x=0])/(Pr3^2)
        // dxpro/di = (P1i*Pr3[i=0]-P3i*Pr1[i=0])/(Pr3^2)
        // dypro/dx = (P21*(P32*y+P33*z+P34)-P31*(P22*y+P23*z+P24))/(Pr3^2) = (P21*Pr3[x=0]-P31*Pr2[x=0])/(Pr3^2)
        // dypro/di = (P2i*Pr3[i=0]-P3i*Pr2[i=0])/(Pr3^2)
        for coord in 0..3 {
            // Create a vector where coord = 0
            let mut vec_diff = point4d;
            vec_diff[coord] = 0.0;
            // Create projection where coord = 0
            let p_r_diff = projection * vec_diff;
            jac[(0, coord)] = (projection[(0, coord)] * p_r_diff[2]
                - projection[(2, coord)] * p_r_diff[0])
                / (p_r[2] * p_r[2]);
            jac[(1, coord)] = (projection[(1, coord)] * p_r_diff[2]
                - projection[(2, coord)] * p_r_diff[1])
                / (p_r[2] * p_r[2]);
        }

        jac
    }

    #[inline]
    fn residual(&self, point_i: usize, view_j: usize) -> Vector2<f64> {
        let point3d = &self.points3d[point_i];
        let projection = &self.projections[view_j];
        let original = if let Some(original) =
            self.tracks[point_i].get_inside_center(view_j, self.image_shapes[view_j])
        {
            original
        } else {
            return Vector2::zeros();
        };
        if let Some(point3d) = point3d {
            let point4d = point3d.insert_row(3, 1.0);
            let mut projected = projection * point4d;
            projected.unscale_mut(projected.z);
            let dx = original.1 as f64 - projected.x;
            let dy = original.0 as f64 - projected.y;

            Vector2::new(dx, dy)
        } else {
            Vector2::zeros()
        }
    }

    fn residual_a(&self, view_j: usize) -> Option<Matrix6x1<f64>> {
        let mut residual = Matrix6x1::zeros();
        for (point_i, point3d) in self.points3d.iter().enumerate() {
            let point3d = (*point3d)?;
            let a = self.jacobian_a(&point3d, view_j);
            let res = self.residual(point_i, view_j);
            residual += a.transpose() * self.covariance * res;
        }
        Some(residual)
    }

    fn residual_b(&self, point_i: usize) -> Option<Vector3<f64>> {
        let mut residual = Vector3::zeros();
        let point3d = &self.points3d[point_i];
        let point3d = (*point3d)?;
        for view_j in 0..self.cameras.len() {
            let b = self.jacobian_b(&point3d, view_j);
            let res = self.residual(point_i, view_j);
            residual += b.transpose() * self.covariance * res;
        }
        Some(residual)
    }

    #[inline]
    fn calculate_u(&self, view_j: usize) -> Option<Matrix6<f64>> {
        let mut u = Matrix6::zeros();
        for point3d in self.points3d.iter() {
            let point3d = (*point3d)?;
            let a = self.jacobian_a(&point3d, view_j);
            u += a.transpose() * self.covariance * a;
        }
        for i in 0..BundleAdjustment::CAMERA_PARAMETERS {
            u[(i, i)] += self.mu;
        }
        Some(u)
    }

    #[inline]
    fn calculate_v_inv(&self, point3d: &Option<Vector3<f64>>) -> Option<Matrix3<f64>> {
        let mut v = Matrix3::zeros();
        let point3d = (*point3d)?;
        for view_j in 0..self.cameras.len() {
            let b = self.jacobian_b(&point3d, view_j);
            v += b.transpose() * self.covariance * b;
        }
        for i in 0..3 {
            v[(i, i)] += self.mu;
        }
        match v.pseudo_inverse(f64::EPSILON) {
            Ok(v_inv) => Some(v_inv),
            Err(_) => None,
        }
    }

    #[inline]
    fn calculate_w(&self, point3d: &Option<Vector3<f64>>, view_j: usize) -> Option<Matrix6x3<f64>> {
        let point3d = (*point3d)?;
        let a = self.jacobian_a(&point3d, view_j);
        let b = self.jacobian_b(&point3d, view_j);
        Some(a.transpose() * self.covariance * b)
    }

    #[inline]
    fn calculate_y(&self, point3d: &Option<Vector3<f64>>, view_j: usize) -> Option<Matrix6x3<f64>> {
        let v_inv = self.calculate_v_inv(point3d)?;
        let w = self.calculate_w(point3d, view_j)?;
        Some(w * v_inv)
    }

    fn calculate_s(&self) -> DMatrix<f64> {
        let mut s = DMatrix::<f64>::zeros(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
        );
        // Divide blocks for parallelization.
        let s_blocks = (0..self.cameras.len())
            .flat_map(|j| (0..self.cameras.len()).map(|k| (j, k)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let s_blocks = s_blocks
            .into_par_iter()
            .map(|(view_j, view_k)| {
                let mut s_jk = Matrix6::zeros();
                if view_j == view_k {
                    let u = self.calculate_u(view_j)?;
                    s_jk += u;
                }
                for point3d in self.points3d.iter() {
                    let y_ij = self.calculate_y(point3d, view_j)?;
                    let w_ik = self.calculate_w(point3d, view_k)?;
                    let y_ij_w = y_ij * w_ik.transpose();
                    s_jk -= y_ij_w;
                }
                Some((view_j, view_k, s_jk))
            })
            .collect::<Vec<_>>();

        s_blocks.iter().for_each(|block| {
            let (j, k, s_jk) = if let Some((j, k, s_jk)) = block {
                (j, k, s_jk)
            } else {
                return;
            };
            s.fixed_view_mut::<6, 6>(
                j * BundleAdjustment::CAMERA_PARAMETERS,
                k * BundleAdjustment::CAMERA_PARAMETERS,
            )
            .copy_from(s_jk);
        });

        s
    }

    fn calculate_e(&self) -> MatrixXx1<f64> {
        let mut e = MatrixXx1::zeros(self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS);

        let e_blocks = (0..self.cameras.len())
            .par_bridge()
            .map(|view_j| {
                let mut e_j = self.residual_a(view_j)?;

                for (i, point3d) in self.points3d.iter().enumerate() {
                    let y_ij = self.calculate_y(point3d, view_j)?;
                    let res = self.residual_b(i)?;
                    e_j -= y_ij * res;
                }
                Some((view_j, e_j))
            })
            .collect::<Vec<_>>();

        e_blocks.iter().for_each(|block| {
            let (j, e_j) = if let Some((j, e_j)) = block {
                (j, e_j)
            } else {
                return;
            };
            e.fixed_view_mut::<6, 1>(j * BundleAdjustment::CAMERA_PARAMETERS, 0)
                .copy_from(e_j);
        });

        e
    }

    fn calculate_delta_b(&self, delta_a: &MatrixXx1<f64>) -> MatrixXx1<f64> {
        let mut delta_b = MatrixXx1::zeros(self.points3d.len() * 3);

        let delta_b_blocks = self
            .points3d
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(point_i, point3d)| {
                let mut residual_b = self.residual_b(point_i)?;

                for view_j in 0..self.cameras.len() {
                    let w_ij = self.calculate_w(point3d, view_j)?;
                    let delta_a_j =
                        delta_a.fixed_view::<6, 1>(view_j * BundleAdjustment::CAMERA_PARAMETERS, 0);
                    residual_b -= w_ij.tr_mul(&delta_a_j);
                }

                let v_inv = self.calculate_v_inv(point3d)?;
                let delta_b_i = v_inv * residual_b;
                Some((point_i, delta_b_i))
            })
            .collect::<Vec<_>>();

        delta_b_blocks.iter().for_each(|block| {
            let (i, delta_b_i) = if let Some((i, delta_b_i)) = block {
                (i, delta_b_i)
            } else {
                return;
            };
            delta_b
                .fixed_view_mut::<3, 1>(i * 3, 0)
                .copy_from(delta_b_i);
        });

        delta_b
    }

    fn calculate_residual_vector(&self) -> MatrixXx1<f64> {
        let mut residuals = MatrixXx1::zeros(self.points3d.len() * self.cameras.len() * 2);

        // TODO: run this in parallel
        for (i, point_i) in self.points3d.iter().enumerate() {
            if point_i.is_none() {
                continue;
            }
            for view_j in 0..self.cameras.len() {
                let residual_b_i = self.residual(i, view_j);
                residuals
                    .fixed_rows_mut::<2>(i * self.cameras.len() * 2 + view_j)
                    .copy_from(&residual_b_i);
            }
        }

        residuals
    }

    fn calculate_jt_residual(&self) -> MatrixXx1<f64> {
        let mut g = MatrixXx1::zeros(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS + self.points3d.len() * 3,
        );

        // TODO: run this in parallel
        // gradient = Jt * residual
        // First 6*m rows of Jt are residuals from camera matrices.
        for (i, point_i) in self.points3d.iter().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };
            for view_j in 0..self.cameras.len() {
                let jac_a_i = self.jacobian_a(point_i, view_j);
                let residual_a_i = self.residual(i, view_j);
                let block = jac_a_i.tr_mul(&residual_a_i);
                let mut target_block =
                    g.fixed_rows_mut::<6>(view_j * BundleAdjustment::CAMERA_PARAMETERS);
                target_block += block;
            }
        }

        // Last 3*n rows of Jt are residuals from point coordinates.
        let mut points_target = g.rows_mut(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
            self.points3d.len() * 3,
        );
        for (i, point_i) in self.points3d.iter().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };
            for view_j in 0..self.cameras.len() {
                let jac_b_i = self.jacobian_b(point_i, view_j);
                let residual_b_i = self.residual(i, view_j);
                let block = jac_b_i.tr_mul(&residual_b_i);
                let mut target_block = points_target.fixed_rows_mut::<3>(i * 3);
                target_block += block;
            }
        }

        g
    }

    fn calculate_delta_step(&self) -> Option<MatrixXx1<f64>> {
        let s = self.calculate_s();
        let e = self.calculate_e();
        let delta_a = s.lu().solve(&e)?;
        let delta_b = self.calculate_delta_b(&delta_a);

        let mut delta = MatrixXx1::zeros(delta_a.len() + delta_b.len());
        delta.rows_mut(0, delta_a.len()).copy_from(&delta_a);
        delta
            .rows_mut(delta_a.len(), delta_b.len())
            .copy_from(&delta_b);

        Some(delta)
    }

    fn update_params(&mut self, delta: &MatrixXx1<f64>) {
        for view_j in 0..self.cameras.len() {
            let camera = &mut self.cameras[view_j];
            camera.r += delta.fixed_rows::<3>(BundleAdjustment::CAMERA_PARAMETERS * view_j);
            camera.t += delta.fixed_rows::<3>(BundleAdjustment::CAMERA_PARAMETERS * view_j + 3);
        }
        self.projections = self
            .cameras
            .iter()
            .map(|camera| camera.projection())
            .collect();

        let points_source = delta.rows(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
            self.points3d.len() * 3,
        );

        for (i, point_i) in self.points3d.iter_mut().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };

            let point_source = points_source.fixed_rows::<3>(i * 3);
            *point_i += Vector3::from(point_source);
        }
    }

    fn optimize(
        &mut self,
    ) -> Result<(), TriangulationError> {
        // Levenberg-Marquardt optimization loop.
        let mut residual = self.calculate_residual_vector();
        let mut jt_residual = self.calculate_jt_residual();

        self.mu = BundleAdjustment::INITIAL_MU;
        let mut nu = 2.0;

        let mut found = false;

        for iter in 0..BUNDLE_ADJUSTMENT_MAX_ITERATIONS {
            if jt_residual.max().abs() <= BundleAdjustment::GRADIENT_EPSILON {
                found = true;
                break;
            }
            let delta = self.calculate_delta_step();
            let delta: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>> = if let Some(delta) = delta {
                delta
            } else {
                return Err(TriangulationError::new("Failed to compute delta vector"));
            };

            let params_norm;
            {
                let sum_cameras = self
                    .cameras
                    .iter()
                    .map(|camera| camera.r.norm_squared() + camera.t.norm_squared())
                    .sum::<f64>();
                let sum_points = self
                    .points3d
                    .iter()
                    .filter_map(|p| Some((*p)?.norm_squared()))
                    .sum::<f64>();
                params_norm = (sum_cameras + sum_points).sqrt();
            }

            if delta.norm()
                <= BundleAdjustment::DELTA_EPSILON * (params_norm + BundleAdjustment::DELTA_EPSILON)
            {
                found = true;
                break;
            }

            let current_cameras = self.cameras.clone();
            let current_projections = self.projections.clone();
            let mut current_points3d = self.points3d.clone();

            self.update_params(&delta);

            let new_residual = self.calculate_residual_vector();
            let residual_norm_squared = residual.norm_squared();
            let new_residual_norm_squared = new_residual.norm_squared();

            let rho = (residual_norm_squared - new_residual_norm_squared)
                / (delta.tr_mul(&(delta.scale(self.mu) + &jt_residual)))[0];

            if rho > 0.0 {
                let converged = residual_norm_squared.sqrt() - new_residual_norm_squared.sqrt()
                    < BundleAdjustment::RESIDUAL_REDUCTION_EPSILON * residual_norm_squared.sqrt();

                residual = new_residual;
                jt_residual = self.calculate_jt_residual();

                if converged || jt_residual.max().abs() <= BundleAdjustment::GRADIENT_EPSILON {
                    found = true;
                    break;
                }
                self.mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powf(3.0));
                nu = 2.0;
            } else {
                self.cameras = current_cameras;
                self.projections = current_projections;
                self.points3d.clear();
                self.points3d.append(&mut current_points3d);
                self.mu *= nu;
                nu *= 2.0;
            }

            if residual.norm() <= BundleAdjustment::RESIDUAL_EPSILON {
                found = true;
                break;
            }
        }

        if !found {
            return Err(TriangulationError::new(
                "Levenberg-Marquardt failed to converge",
            ));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct TriangulationError {
    msg: &'static str,
}

impl TriangulationError {
    fn new(msg: &'static str) -> TriangulationError {
        TriangulationError { msg }
    }
}

impl std::error::Error for TriangulationError {}

impl fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
