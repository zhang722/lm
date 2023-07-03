use std::{ collections::HashMap, cell::RefCell, rc::Rc };

use nalgebra as na;

pub trait Vertex {
    fn id(&self) -> usize;
    fn id_mut(&mut self) -> &mut usize;
    fn edges(&self) -> &Vec<usize>;
    fn edges_mut(&mut self) -> &mut Vec<usize>;
    fn params(&self) -> &na::DVector<f64>;
    fn plus(&mut self, delta: &na::DVector<f64>);

    fn dimension(&self) -> usize {
        self.params().len()
    }
    fn add_edge(&mut self, id: usize) {
        self.edges_mut().push(id);
    }
}

type VertexBase = Rc<RefCell<dyn Vertex>>;

pub trait Edge {
    fn id(&self) -> usize;
    fn id_mut(&mut self) -> &mut usize;
    fn vertex(&self, ith: usize) -> VertexBase;
    fn vertices(&self) -> &Vec<VertexBase>;
    fn vertices_mut(&mut self) -> &mut Vec<VertexBase>;
    fn residual(&self) -> na::DVector<f64>;
    fn jacobain(&self, ith: usize) -> na::DMatrix<f64>;
    fn sigma(&self) -> na::DMatrix<f64>;

    fn dimension(&self) -> usize {
        self.residual().len()
    }
    fn add_vertex(&mut self, vertex: VertexBase) {
        self.vertices_mut().push(vertex.clone());
        vertex.borrow_mut().add_edge(self.id());
    }
}

type EdgeBase = Rc<RefCell<dyn Edge>>;

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
    pub fn default() -> Self {
        Self::new(1e-3, 1e-15, 1e-15, 1e-15, 2.0, 100)
    }
}

pub struct Graph {
    vertices: Vec<Vec<VertexBase>>,
    edges: HashMap<usize, EdgeBase>,
    lm_params: LmParams,
}

impl Graph {
    pub fn default() -> Self {
        Self {
            vertices: Vec::new(),
            edges: HashMap::<usize, EdgeBase>::new(),
            lm_params: LmParams::default(),
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

    pub fn update_params(&mut self, delta: &na::DVector<f64>) {
        todo!()
    }

    pub fn vertex2param(&self) -> na::DVector<f64> {
        todo!()
    }

    pub fn params_norm(&self) -> f64 {
        self.vertex2param().norm() 
    }

    fn init_u(&self) -> f64 {
        todo!()
    }

    pub fn calculate_residual(&self) -> na::DVector<f64> {
        let residual_len = self.edges.iter().fold(0usize, |acc, (_, x)| {
            acc + x.borrow().dimension()
        });

        let mut res = na::DVector::<f64>::zeros(residual_len);

        let mut idx = 0;
        for (_id, edge) in self.edges.iter() {
            let edge_len = edge.borrow().dimension();
            res.rows_mut(idx, edge_len).copy_from(
                &edge.borrow().residual()
            );
            idx += edge_len; 
        }

        res
    }

    // pub fn calculate_jt_residual(&self) -> na::DVector<f64> {
    //     let jt_residual_a_len = self.vertices[0].iter().fold(0usize, |acc, x| {
    //         acc + x.dimension()
    //     });
    //     let jt_residual_b_len = self.vertices[1].iter().fold(0usize, |acc, x| {
    //         acc + x.dimension()
    //     });

    //     let mut jt_residual = na::DVector::<f64>::zeros(jt_residual_a_len + jt_residual_b_len);
        
    //     let mut idx = 0;
    //     for vertex in self.vertices[0].iter() {
    //         let dim = vertex.dimension();
    //         let mut jt_residual_aj = na::DVector::<f64>::zeros(dim);
    //         for edge in self.edges.iter() {
    //             if !Rc::ptr_eq(&edge.vertex0(), &vertex) {
    //                 continue;
    //             }

    //             jt_residual_aj += edge.jacobian0().transpose() * edge.sigma() * edge.residual();
    //         }
    //         jt_residual.rows_mut(idx, dim).copy_from(
    //             &jt_residual_aj
    //         );
    //         idx += dim;
    //     }

    //     for vertex in self.vertices[1].iter() {
    //         let dim = vertex.dimension();
    //         let mut jt_residual_ai = na::DVector::<f64>::zeros(dim);
    //         for edge in self.edges.iter() {
    //             if !Rc::ptr_eq(&edge.vertex1(), &vertex) {
    //                 continue;
    //             }

    //             jt_residual_ai += edge.jacobian1().transpose() * edge.sigma() * edge.residual();
    //         }
    //         jt_residual.rows_mut(idx, dim).copy_from(
    //             &jt_residual_ai
    //         );
    //         idx += dim;
    //     }

    //     jt_residual
    // }

    // #[inline]
    // fn calculate_u(&self, vertex0_j: usize) -> na::DMatrix<f64> {
    //     let dim = self.vertices[0][vertex0_j].dimension();
    //     let mut u = self.lm_params.u * na::DMatrix::<f64>::identity(dim, dim);
    //     for edge in self.edges.iter() {
    //         if !Rc::ptr_eq(&edge.vertex0(), &self.vertices[0][vertex0_j]) {
    //             continue;
    //         }

    //         u += edge.jacobian0().transpose() * edge.sigma() * edge.jacobian0();
    //     }

    //     u
    // }

    // #[inline]
    // fn calculate_v_inv(&self, vertex1_i: usize) -> Option<na::DMatrix<f64>> {
    //     let dim = self.vertices[1][vertex1_i].dimension();
    //     let mut v = self.lm_params.u * na::DMatrix::<f64>::identity(dim, dim);
    //     for edge in self.edges.iter() {
    //         if !Rc::ptr_eq(&edge.vertex1(), &self.vertices[1][vertex1_i]) {
    //             continue;
    //         }

    //         v += edge.jacobian1().transpose() * edge.sigma() * edge.jacobian1();
    //     }

    //     match v.pseudo_inverse(f64::EPSILON) {
    //         Ok(v_inv) => Some(v_inv),
    //         Err(_) => None,
    //     }
    // }

    // #[inline]
    // fn calculate_w(&self, vertex0_j: usize, vertex1_i: usize) -> na::DMatrix<f64> {
    //     let nrows = self.vertices[0][vertex0_j].dimension();
    //     let ncols = self.vertices[1][vertex1_i].dimension();
    //     let mut w = na::DMatrix::<f64>::zeros(nrows, ncols);
    //     for edge in self.edges.iter() {
    //         if !Rc::ptr_eq(&edge.vertex0(), &self.vertices[0][vertex0_j]) ||
    //             !Rc::ptr_eq(&edge.vertex1(), &self.vertices[1][vertex1_i]) {
    //             continue;
    //         }

    //         w += edge.jacobian0().transpose() * edge.sigma() * edge.jacobian1();
    //     }

    //     w
    // }

    // #[inline]
    // fn calculate_y(&self, vertex0_j: usize, vertex1_i: usize) -> Option<na::DMatrix<f64>> {
    //     Some(self.calculate_w(vertex0_j, vertex1_i) * self.calculate_v_inv(vertex1_i)?)
    // }

    // fn calculate_s(&self) -> Option<na::DMatrix<f64>> {
    //     let dim = self.vertices[0].iter().fold(0usize, |acc, x| {
    //         acc + x.dimension()
    //     });

    //     let mut s = na::DMatrix::<f64>::zeros(
    //         dim,
    //         dim
    //     );
        
    //     // Divide blocks for parallelization.
    //     let s_blocks = (0..self.vertices[0].len())
    //         .flat_map(|j| (0..self.vertices[0].len()).map(|k| (j, k)).collect::<Vec<_>>())
    //         .collect::<Vec<_>>();

    //     let s_blocks = s_blocks
    //         .into_iter()
    //         .map(|(view_j, view_k)| {
    //             let mut s_jk = na::DMatrix::<f64>::zeros(
    //                 self.vertices[0][view_j].dimension(),
    //                 self.vertices[0][view_k].dimension(),
    //             );
    //             if view_j == view_k {
    //                 let u = self.calculate_u(view_j);
    //                 s_jk += u;
    //             }
    //             for (vertex1_i, vertex ) in self.vertices[1].iter().enumerate() {
    //                 let y_ij = self.calculate_y(vertex1_i, view_j)?;
    //                 let w_ik = self.calculate_w(vertex1_i, view_k);
    //                 let y_ij_w = y_ij * w_ik.transpose();
    //                 s_jk -= y_ij_w;
    //             }
    //             Some((view_j, view_k, s_jk))
    //         })
    //         .collect::<Vec<_>>();

    //     s_blocks.iter().for_each(|block| {
    //         let (j, k, s_jk) = if let Some((j, k, s_jk)) = block {
    //             (j, k, s_jk)
    //         } else {
    //             return;
    //         };
    //         s.view_mut(
    //             (j * self.vertices[0][0].dimension(), k * self.vertices[0][0].dimension()),
    //             (self.vertices[0][0].dimension(), self.vertices[0][0].dimension()),
    //         )
    //         .copy_from(s_jk);
    //     });

    //     Some(s)
    // }

    // fn calculate_e(&self) -> Option<na::DVector<f64>> {
    //     let dim = self.vertices[0].iter().fold(0usize, |acc, x| {
    //         acc + x.dimension()
    //     });
    //     let mut e = na::DVector::zeros(dim);

    //     let e_blocks = self.vertices[0].iter().enumerate()
    //         .map(|(vertex0_j, vertex0)| {
    //             let dim0 = vertex0.dimension();
    //             let mut e_j = na::DVector::<f64>::zeros(dim0);
    //             for (vertex1_i, vertex1 ) in self.vertices[1].iter().enumerate() {
    //                 let dim1 = vertex1.dimension();
    //                 let mut jt_residual_bi = na::DVector::<f64>::zeros(dim1);
    //                 for edge in self.edges.iter() {
    //                     if !Rc::ptr_eq(&edge.vertex0(), vertex0) || 
    //                         !Rc::ptr_eq(&edge.vertex1(), vertex1) {
    //                         continue;
    //                     } 
    //                     jt_residual_bi += edge.jacobian1() * edge.sigma() * edge.residual();
    //                 }
    //                 for edge in self.edges.iter() {
    //                     if !Rc::ptr_eq(&edge.vertex0(), vertex0) || 
    //                         !Rc::ptr_eq(&edge.vertex1(), vertex1) {
    //                         continue;
    //                     } 
    //                     e_j += edge.jacobian0().transpose() * edge.sigma() * edge.residual();
    //                     e_j -= self.calculate_y(vertex0_j, vertex1_i)? * &jt_residual_bi;
    //                 }
    //             }
    //             Some((vertex0_j, e_j))
    //         })
    //         .collect::<Vec<_>>();

    //     let mut idx = 0;
    //     e_blocks.iter().for_each(|block| {
    //         let (j, e_j) = if let Some((j, e_j)) = block {
    //             (j, e_j)
    //         } else {
    //             return;
    //         };
    //         let dim = self.vertices[0][*j].dimension();
    //         e.rows_mut(idx, dim)
    //             .copy_from(e_j);
    //         idx += dim;
    //     });

    //     Some(e)
    // }

    // fn calculate_delta_b(&self, delta_a: &na::DVector<f64>) -> Option<na::DVector<f64>> {
    //     let dim = self.vertices[0].iter().fold(0usize, |acc, x| {
    //         acc + x.dimension()
    //     });
    //     let mut e = na::DVector::zeros(dim);

    //     let e_blocks = self.vertices[1].iter().enumerate()
    //         .map(|(vertex1_i, vertex1)| {
    //             let dim1 = vertex1.dimension();
    //             let mut idx = 0; 
    //             let mut e_i = na::DVector::<f64>::zeros(dim1);
    //             for (vertex0_j, vertex0 ) in self.vertices[0].iter().enumerate() {
    //                 let dim0 = vertex0.dimension();
    //                 let delta_aj = delta_a.rows(idx, dim0).clone_owned();
    //                 idx += dim0;
    //                 for edge in self.edges.iter() {
    //                     if !Rc::ptr_eq(&edge.vertex0(), vertex0) || 
    //                         !Rc::ptr_eq(&edge.vertex1(), vertex1) {
    //                         continue;
    //                     } 
    //                     e_i += edge.jacobian1() * edge.sigma() * edge.residual();
    //                     e_i -= self.calculate_w(vertex0_j, vertex1_i).transpose() * &delta_aj;
    //                 }
    //             }
    //             Some((vertex1_i, e_i))
    //         })
    //         .collect::<Vec<_>>();

    //     let mut idx = 0;
    //     e_blocks.iter().for_each(|block| {
    //         let (i, e_i) = if let Some((i, e_i)) = block {
    //             (i, e_i)
    //         } else {
    //             return;
    //         };
    //         let dim = self.vertices[1][*i].dimension();
    //         e.rows_mut(idx, dim)
    //             .copy_from(e_i);
    //         idx += dim;
    //     });

    //     Some(e)

    // }

    // pub fn calculate_delta_step(&self) -> Option<na::DVector<f64>> {
    //     let s = self.calculate_s()?;
    //     let e = self.calculate_e();
    //     let delta_a = s.lu().solve(&e?).unwrap();
    //     let delta_b = self.calculate_delta_b(&delta_a)?;

    //     let mut delta = na::DVector::<f64>::zeros(delta_a.len() + delta_b.len());
    //     delta.rows_mut(0, delta_a.len()).copy_from(&delta_a);
    //     delta
    //         .rows_mut(delta_a.len(), delta_b.len())
    //         .copy_from(&delta_b);

    //     Some(delta)
    // }

    // pub fn optimize(&mut self) -> Option<()>{
    //     let mut v = self.lm_params.v;
    //     let mut e = self.calculate_residual();
    //     let mut jt_residual = self.calculate_jt_residual();
    //     let mut stop = jt_residual.abs().max() < self.lm_params.eps1;
    //     let mut k = 0;

    //     while k < self.lm_params.max_iter && !stop {
    //         k += 1;
    //         let rho = 0.0;
    //         while rho <= 0.0 && !stop {
    //             if k == 1 {
    //                 self.init_u();
    //             }
    //             let delta = self.calculate_delta_step()?;
    //             if delta.norm() <= self.lm_params.eps2 * self.params_norm() {
    //                 stop = true;
    //             } else {
    //                 self.update_params(&delta);
    //                 let e1 = self.calculate_residual();
    //                 let rho = (e.norm().powi(2) - e1.norm().powi(2)) / (delta.transpose() * (self.lm_params.u * delta - &jt_residual))[0];
    //                 if rho > 0.0 {
    //                     e = e1;
    //                     jt_residual = self.calculate_jt_residual();
    //                     stop = jt_residual.abs().max() < self.lm_params.eps1 || e.norm() < self.lm_params.eps3;
    //                     self.lm_params.u *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
    //                     v = 2.0;
    //                 } else {
    //                     self.lm_params.u *= v;
    //                     v *= 2.0;
    //                 }   
    //             }
    //         }
    //     }

    //     Some(())
    // }
}

// pub struct CameraVertex {
//     id: usize,
//     edges: Vec<usize>,
//     params: na::DVector<f64>,
// }

// impl Vertex for CameraVertex {
//     fn id(&self) -> usize {
//         self.id
//     }

//     fn edges(&self) -> Vec<usize> {
//         self.edges        
//     }

//     fn dimension(&self) -> usize {
//         9
//     }

//     fn params(&self) -> &na::DVector<f64> {
//         &self.params
//     }

//     fn plus(&mut self, delta: &na::DVector<f64>) {
//         self.params += delta;
//     }
// }

// pub struct PointVertex {
//     id: usize,
//     edges: Vec<usize>,
//     params: na::DVector<f64>,
// }

// impl Vertex for PointVertex {
//     fn id(&self) -> usize {
//         self.id
//     }

//     fn edges(&self) -> Vec<usize> {
//         self.edges
//     }

//     fn dimension(&self) -> usize {
//         3
//     }

//     fn params(&self) -> &na::DVector<f64> {
//         &self.params
//     }

//     fn plus(&mut self, delta: &na::DVector<f64>) {
//         self.params += delta;
//     }
// }

// pub struct Point3dProjectEdge {
//     id: usize,
//     vertices: Vec<usize>,
//     sigma: na::DMatrix<f64>,
//     measurement: na::DVector<f64>,
// }

// type CameraInstrinsics = na::Vector3<f64>;

// fn jacobian_pp_wrt_pn(
//     pn: &na::Vector2<f64>, 
//     intrinsics: &CameraInstrinsics,
// ) -> na::Matrix2<f64> 
// {
//     let f = intrinsics[0];
//     let k1 = intrinsics[1];
//     let k2 = intrinsics[2];
//     let x = pn[0];
//     let y = pn[1];

//     let rn2 = x.powi(2) + y.powi(2);
//     let rn4 = rn2.powi(2);

//     na::Matrix2::<f64>::new(
//         f * (k1 * rn2 + k2 * rn4 + 1.0), 0.0,
//         0.0, f * (k1 * rn2 + k2 * rn4 + 1.0)
//     )
// }

// fn jacobian_pn_wrt_ps(
//     ps: &na::Vector3<f64>,
// ) -> na::Matrix2x3<f64>
// {
//     let x = ps[0];
//     let y = ps[1];
//     let z = ps[2];
//     let z2 = z.powi(2);

//     -na::Matrix2x3::<f64>::new(
//         1.0 / z, 0.0, -x / z2, 
//         0.0, 1.0 / z, -y / z2)
// }

// /// Produces a skew-symmetric or "cross-product matrix" from
// /// a 3-vector. This is needed for the `exp_map` and `log_map`
// /// functions
// fn skew_sym(v: na::Vector3<f64>) -> na::Matrix3<f64> {
//     let mut ss = na::Matrix3::zeros();
//     ss[(0, 1)] = -v[2];
//     ss[(0, 2)] = v[1];
//     ss[(1, 0)] = v[2];
//     ss[(1, 2)] = -v[0];
//     ss[(2, 0)] = -v[1];
//     ss[(2, 1)] = v[0];
//     ss
// }

// /// Converts an NAlgebra Isometry to a 6-Vector Lie Algebra representation
// /// of a rigid body transform.
// ///
// /// This is largely taken from this paper:
// /// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
// fn log_map(input: &na::Isometry3<f64>) -> na::Vector6<f64> {
//     let t: na::Vector3<f64> = input.translation.vector;

//     let quat = input.rotation;
//     let theta: f64 = 2.0 * (quat.scalar()).acos();
//     let half_theta = 0.5 * theta;
//     let mut omega = na::Vector3::<f64>::zeros();

//     let mut v_inv = na::Matrix3::<f64>::identity();
//     if theta > 1e-6 {
//         omega = quat.vector() * theta / (half_theta.sin());
//         let ssym_omega = skew_sym(omega);
//         v_inv -= ssym_omega * 0.5;
//         v_inv += ssym_omega * ssym_omega * (1.0 - half_theta * half_theta.cos() / half_theta.sin())
//             / (theta * theta);
//     }

//     let mut ret = na::Vector6::<f64>::zeros();
//     ret.fixed_view_mut::<3, 1>(0, 0).copy_from(&(v_inv * t));
//     ret.fixed_view_mut::<3, 1>(3, 0).copy_from(&omega);

//     ret
// }

// fn jacobian_ps_wrt_pose(
//     ps: &na::Vector3<f64>,
// ) -> na::Matrix3x6<f64>
// {
//     let x = ps[0];
//     let y = ps[1];
//     let z = ps[2];

//     let mut jac = na::Matrix3x6::<f64>::zeros();
//     jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&na::Matrix3::<f64>::identity());
//     jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&-skew_sym(na::Vector3::<f64>::new(x, y, z)));
//     jac
// }

// fn jacobian_ps_wrt_pw(
//     Rsw: &na::Rotation3<f64>,
// ) -> na::Matrix3<f64>
// {
//     let mut jac = na::Matrix3::<f64>::zeros();
//     jac.copy_from(Rsw.matrix());
//     jac
// }

// fn jacobian_pp_wrt_intrinsics(
//     pn: &na::Vector2<f64>,
//     intrinsics: &CameraInstrinsics,
// ) -> na::Matrix2x3<f64>
// {
//     let f = intrinsics[0];
//     let k1 = intrinsics[1];
//     let k2 = intrinsics[2];
//     let x = pn[0];
//     let y = pn[1];

//     let rn2 = x.powi(2) + y.powi(2);
//     let rn4 = rn2.powi(2);

//     na::Matrix2x3::<f64>::new(
//         x * (k1 * rn2 + k2 * rn4 + 1.0), f * rn2 * x, f * rn4 * x,
//         y * (k1 * rn2 + k2 * rn4 + 1.0), f * rn2 * y, f * rn4 * y
//     )
// }

// impl Edge for Point3dProjectEdge {
//     fn dimension(&self) -> usize {
//         2        
//     }

//     fn residual(&self) -> na::DVector<f64> {
//         let camera = &self.vertex0.as_ref().params;
//         let point3d = &self.vertex1.as_ref().params;

//         let rotation = na::Rotation3::new(
//             na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
//         );
//         let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
//         let f = camera[6];
//         let k1 = camera[7];
//         let k2 = camera[8];

//         let ps = rotation * point3d + t;
//         let pn = -ps / ps.z;
//         let pn = pn.fixed_view::<2, 1>(0, 0);
//         let pp = f * (1.0 + k1 * pn.norm().powi(2) + k2 * pn.norm().powi(4)) * pn;
//         let mut res = na::DVector::zeros(pp.len());
//         res.fixed_view_mut::<2, 1>(0, 0).copy_from(
//             &(pp - na::Vector2::new(self.measurement[0], self.measurement[1]))
//         );

//         res
//     }

//     fn jacobian0(&self) -> na::DMatrix<f64> {
//         let camera = &self.vertex0.as_ref().params;
//         let point3d = &self.vertex1.as_ref().params;

//         let mut jac = na::DMatrix::zeros(2, camera.len());

//         let rotation = na::Rotation3::new(
//             na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
//         );
//         let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
//         let intrinsics = na::Vector3::<f64>::new(camera[6], camera[7], camera[8]);

//         let ps = rotation * point3d + t;
//         let pn = -ps / ps.z;
//         let pn = pn.fixed_view::<2, 1>(0, 0).clone_owned();

//         let jacobian_r_wrt_pose = jacobian_pp_wrt_pn(&pn, &intrinsics) 
//             * jacobian_pn_wrt_ps(&ps) 
//             * jacobian_ps_wrt_pose(&ps);
//         let jacobian_r_wrt_intrinsics = jacobian_pp_wrt_intrinsics(&pn, &intrinsics);
        
//         jac.fixed_view_mut::<2, 6>(0, 0)
//             .copy_from(&jacobian_r_wrt_pose);
//         jac.fixed_view_mut::<2, 3>(0, 6)
//             .copy_from(&jacobian_r_wrt_intrinsics);
        
//         jac
//     }

//     fn jacobian1(&self) -> na::DMatrix<f64> {
//         let camera = &self.vertex0.as_ref().params;
//         let point3d = &self.vertex1.as_ref().params;

//         let mut jac = na::DMatrix::zeros(2, point3d.len());

//         let rotation = na::Rotation3::new(
//             na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
//         );
//         let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
//         let intrinsics = na::Vector3::<f64>::new(camera[6], camera[7], camera[8]);

//         let ps = rotation * point3d + t;
//         let pn = -ps / ps.z;
//         let pn = pn.fixed_view::<2, 1>(0, 0).clone_owned();

//         let jacobian_r_wrt_pw = jacobian_pp_wrt_pn(&pn, &intrinsics) 
//             * jacobian_pn_wrt_ps(&ps) 
//             * jacobian_ps_wrt_pw(&rotation);
//         jac.fixed_view_mut::<2, 3>(0, 0)
//             .copy_from(&jacobian_r_wrt_pw); 

//         jac
//     }

//     fn sigma(&self) -> na::DMatrix<f64> {
//         na::DMatrix::<f64>::identity(2, 2)
//     }

//     fn id(&self) -> usize {
//         self.id
//     }

//     fn vertices(&self) -> Vec<usize> {
//         self.vertices
//     }
// }

mod tests{
    use std::{rc::Rc, cell::RefCell};

    use nalgebra as na;

    use super::{Edge, Vertex};
    struct CameraVertex {
        id: usize,
        params: na::DVector<f64>,
        edges: Vec<usize>,
    }

    impl super::Vertex for CameraVertex {
        fn id(&self) -> usize {
            self.id
        }

        fn id_mut(&mut self) -> &mut usize {
            &mut self.id
        }

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
        }
    }

    struct PointVertex {
        id: usize,
        params: na::DVector<f64>,
        edges: Vec<usize>,
    }

    impl super::Vertex for PointVertex {
        fn id(&self) -> usize {
            self.id
        }

        fn id_mut(&mut self) -> &mut usize {
            &mut self.id
        }

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
        }
    }

    struct ProjectEdge {
        id: usize,
        vertices: Vec<super::VertexBase>,
        sigma: na::DMatrix<f64>,
        measurement: na::DVector<f64>,        
    }

    impl super::Edge for ProjectEdge {
        fn id(&self) -> usize {
            self.id
        }

        fn id_mut(&mut self) -> &mut usize {
            &mut self.id
        }

        fn vertex(&self, ith: usize) -> super::VertexBase {
            self.vertices[ith].clone()
        }

        fn vertices(&self) -> &Vec<super::VertexBase> {
            &self.vertices
        }

        fn vertices_mut(&mut self) -> &mut Vec<super::VertexBase> {
            &mut self.vertices
        }

        fn residual(&self) -> na::DVector<f64> {
            na::DVector::<f64>::zeros(self.measurement.len())
        }

        fn jacobain(&self, ith: usize) -> na::DMatrix<f64> {
            na::DMatrix::<f64>::zeros(self.measurement.len(), self.vertices[ith].borrow().dimension())
        }

        fn sigma(&self) -> na::DMatrix<f64> {
            na::DMatrix::<f64>::zeros(self.measurement.len(), self.measurement.len())
        }
    }

    #[test]
    fn test_graph() {
        fn create(id: usize, params: na::DVector<f64>, edges: Vec<usize>) -> super::VertexBase {
            Rc::new(RefCell::new(CameraVertex {
                id, params, edges
            }))
        }
        let camera_vertices: Vec<super::VertexBase> = (0..3usize).into_iter().map(|x| {
            // Rc::new(RefCell::new(CameraVertex {
            //     id: x,
            //     params: na::DVector::<f64>::zeros(9),
            //     edges: Vec::new(),
            // })) as super::VertexBase
            create(x, na::DVector::<f64>::zeros(9), Vec::new())
        }).collect::<Vec<super::VertexBase>>();
        // let camera_vertex0: Rc<RefCell<dyn Vertex>> = Rc::new(RefCell::new(
        //     CameraVertex {
        //         id: 0,
        //         params: na::DVector::<f64>::zeros(9),
        //         edges: Vec::new(),
        //     }
        // ));
        // let camera_vertices = vec![camera_vertex0];

        let point_vertices = (0..10usize).into_iter().map(|x| {
            Rc::new(RefCell::new(PointVertex {
                id: x + 3,
                params: na::DVector::<f64>::zeros(3),
                edges: Vec::new(),
            })) as super::VertexBase
        }).collect::<Vec<_>>();

        let edge1: Rc<RefCell<dyn super::Edge>> = Rc::new(RefCell::new(ProjectEdge {
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
    }
}