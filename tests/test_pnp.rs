use std::error::Error;
use std::rc::Rc;
use std::cell::RefCell;

use nalgebra as na;

use lm::*;

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
fn log_map(input: &na::Isometry3<f64>) -> na::DVector<f64> {
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

    let mut ret = na::DVector::<f64>::zeros(6);
    ret.fixed_view_mut::<3, 1>(0, 0).copy_from(&(v_inv * t));
    ret.fixed_view_mut::<3, 1>(3, 0).copy_from(&omega);

    ret
}

#[cfg(test)]
pub fn load(path: &str) -> Result<Graph, Box<dyn Error>>  {
    // read path file to string

    use na::dvector;
    let content = std::fs::read_to_string(path)?;

    let v: serde_json::Value = serde_json::from_str(&content)?;
    let mut points3d = Vec::new();
    let mut poses = Vec::new();
    for p in v["points"].as_array().unwrap() {
        points3d.push(na::DVector::<f64>::from_vec(vec![p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap()]));
    }
    let intrinsics = CameraInstrinsics::from_vec(vec![458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]);
    for p in points3d.iter().take(3) {
        let pdn = ps2pdn(&intrinsics, &na::Vector3::new(p[0], p[1], p[2]));
        let pp = pdn2pp(&intrinsics, &pdn);
        println!("{}", pp);
    }

    let mut graph = Graph::default();
    let mut id = 0;
    let point_vertices = points3d.iter().map(|x| {
        let ret = Rc::new(RefCell::new(PointVertex {
            id,
            params: x.clone(),
            edges: Vec::new(),
            fixed: false,
            hessian_index: 0,
        })) as VertexBase;
        id += 1;
        ret
    }).collect::<Vec<_>>();

    for f in v["keyframes"].as_array().unwrap() {
        let mut pt2d = Vec::new();
        let mut camera_model = Vec::new();
        for p in f["points"].as_array().unwrap() {
            pt2d.push(na::Point2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()));
        }
        for param in f["camera_model"].as_array().unwrap() {
            camera_model.push(param.as_f64().unwrap());
        }
        poses.push(log_map(&na::Isometry3::from_parts(
            na::Translation3::new(
                f["pose"]["x"].as_f64().unwrap(),
                f["pose"]["y"].as_f64().unwrap(),
                f["pose"]["z"].as_f64().unwrap(),
            ),
            na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                    f["pose"]["q"][3].as_f64().unwrap(),
                    f["pose"]["q"][0].as_f64().unwrap(),
                    f["pose"]["q"][1].as_f64().unwrap(),
                    f["pose"]["q"][2].as_f64().unwrap(),
            )),
        ))); 
    }

    let camera_vertices = poses.iter().map(|x| {
        let ret = Rc::new(RefCell::new(CameraVertex {
            id,
            params: x.clone(),
            edges: Vec::new(),
            fixed: false,
            hessian_index: 0,
        })) as VertexBase;
        id += 1;
        ret
    }).collect::<Vec<_>>();

    id = 0;
    for (camera_idx, f) in v["keyframes"].as_array().unwrap().iter().enumerate() {
        for (point_idx, p) in f["points"].as_array().unwrap().iter().enumerate() {
            let edge = Rc::new(RefCell::new( Point3dProjectWithIntrinsicEdge {
                id,
                vertices: Vec::new(),
                sigma: na::DMatrix::<f64>::identity(2, 2),
                measurement: dvector![p[0].as_f64().unwrap(), p[1].as_f64().unwrap()],
                intrinsic: CameraInstrinsics::from_vec(vec![458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]),
            }
            )) as EdgeBase;
            id += 1;
            edge.borrow_mut().add_vertex(camera_vertices[camera_idx].clone());
            edge.borrow_mut().add_vertex(point_vertices[point_idx].clone());
            graph.add_edge(&edge);
        }

    }
    graph.add_vertex_set(camera_vertices);
    graph.add_vertex_set(point_vertices);

    Ok(graph)
}


#[cfg(test)]
#[test]
fn test_position() {
    let mut graph: Graph = load("scene.json").unwrap();
    // graph.optimize();
    // graph.print();
}