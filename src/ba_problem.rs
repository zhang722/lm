use std::cell::RefCell;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra as na;
use rand::Rng;

#[derive(Debug)]
pub struct Observation {
    point_idx: usize,
    camera_idx: usize,
    x: f64,
    y: f64,
    x_std: f64,
    y_std: f64,
}

pub struct BaProblem {
    num_cameras: usize,
    observations: Vec<Observation>,
    parameters: na::DVector<f64>,
    cameras: Vec<na::DVector<f64>>,
    points: Vec<na::DVector<f64>>,
}

impl BaProblem {
    pub fn new(filename: &str, max_num_cameras: usize, max_num_points: usize) -> Self {
        // parse file 
        let file = File::open(filename).expect("open file failed!");
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let (num_cameras, num_points, num_observations) = match lines.next() {
            Some(line) => {
                let line = line.expect("line read failed!");
                let mut iter = line.split_whitespace();
                let num_cameras = iter.next().unwrap().parse::<usize>().unwrap();
                let num_points = iter.next().unwrap().parse::<usize>().unwrap();
                let num_observations = iter.next().unwrap().parse::<usize>().unwrap();

                (num_cameras, num_points, num_observations)
            },
            None => panic!("file is empty!"),
        };

        let expected_num_cameras = if num_cameras > max_num_cameras {
            max_num_cameras   
        } else {
            num_cameras
        };
        let expected_num_points = if num_points > max_num_points {
            max_num_points   
        } else {
            num_points
        };

        let mut observations = Vec::with_capacity(num_observations);
        let mut parameters_cameras = Vec::with_capacity(9 * expected_num_cameras);
        let mut parameters_points = Vec::with_capacity(3 * expected_num_points);

        let parse_observation = |line: &str| -> Option<Observation> {
            let mut split = line.split_whitespace();
            let camera_idx = split.next().unwrap().parse::<usize>().unwrap();
            let point_idx = split.next().unwrap().parse::<usize>().unwrap();
            if camera_idx >= expected_num_cameras || point_idx >= expected_num_points {
                return None;
            }

            let x = split.next().unwrap().parse::<f64>().unwrap();
            let y = split.next().unwrap().parse::<f64>().unwrap();
            Some(Observation {
                point_idx,
                camera_idx,
                x,
                y,
                x_std: 0.0,
                y_std: 0.0,
            })           
        };

        for (line_idx, line) in lines.enumerate() {
            let line = line.expect("line read failed!");
            if line_idx < num_observations {
                let observation = match parse_observation(&line) {
                    Some(observation) => observation,
                    None => continue,
                };
                observations.push(observation);
            } else if line_idx < num_observations + 9 * num_cameras {
                let e = line.parse::<f64>().unwrap();
                if parameters_cameras.len() < parameters_cameras.capacity() {
                    parameters_cameras.push(e);
                }
            } else {
                let e = line.parse::<f64>().unwrap();
                if parameters_points.len() < parameters_points.capacity() {
                    parameters_points.push(e);
                }
            }
        }

        let mut cameras = Vec::new();
        for camera in parameters_cameras.chunks(9) {
            cameras.push(na::DVector::from_column_slice(camera));
        }
        let mut points = Vec::new();
        for point in parameters_points.chunks(3) {
            points.push(na::DVector::from_column_slice(point));
        }

        parameters_cameras.append(&mut parameters_points);

        Self {
            num_cameras: expected_num_cameras,
            observations,
            parameters: na::DVector::from_vec(parameters_cameras),
            cameras,
            points,
        }
    }

    fn get_camera_index(&self, camera_idx: usize) -> usize {
        9 * camera_idx
    }

    fn get_point_index(&self, point_idx: usize) -> usize {
        9 * self.num_cameras + 3 * point_idx
    }
}

use std::fs::OpenOptions;
use std::io::prelude::*;
use ply_rs::ply::{ Ply, DefaultElement, Encoding, ElementDef, PropertyDef, PropertyType, ScalarType, Property, Addable };
use ply_rs::writer::{ Writer };
fn savePly(filename: &str, vertices: &[f64]) {
    // crete a ply objet
    let mut ply = {
        let mut ply = Ply::<DefaultElement>::new();
        ply.header.encoding = Encoding::Ascii;
        ply.header.comments.push("A beautiful comment!".to_string());

        // Define the elements we want to write. In our case we write a 2D Point.
        // When writing, the `count` will be set automatically to the correct value by calling `make_consistent`
        let mut point_element = ElementDef::new("point".to_string());
        let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        ply.header.elements.add(point_element);

        // Add data
        let mut points = Vec::new();

        for v in vertices.chunks(3) {
            // Add first point
            let mut point = DefaultElement::new();
            point.insert("x".to_string(), Property::Float(v[0] as f32));
            point.insert("y".to_string(), Property::Float(v[1] as f32));
            point.insert("z".to_string(), Property::Float(v[2] as f32));
            points.push(point);
        }

        ply.payload.insert("point".to_string(), points);

        // only `write_ply` calls this by itself, for all other methods the client is
        // responsible to make the data structure consistent.
        // We do it here for demonstration purpose.
        ply.make_consistent().unwrap();
        ply
    };

    let mut file_result = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(filename).unwrap();

    // set up a writer
    let w = Writer::new();
    let written = w.write_ply(&mut file_result, &mut ply).unwrap();
}    

#[test]
fn test_ba_problem() {
    use crate::graph::*;

    let num_camera = 10;
    let num_point = 5000;
    let problem = BaProblem::new("problem-16-22106-pre.txt", num_camera, num_point);
    // let problem = BaProblem::generate();
    println!("num_observations: {}", problem.observations.len());
    println!("num_camera: {}", problem.cameras.len());
    println!("params: {}", problem.parameters.len());
    savePly("output.ply", &problem.parameters.as_slice()[9 * num_camera ..]);

    let mut graph = crate::graph::Graph::default();
    let mut id = 0;
    let camera_vertices = problem.cameras.iter().map(|x| {
        id += 1;
        Rc::new(RefCell::new(CameraVertex {
            id,
            params: x.clone(),
            edges: Vec::new(),
            fixed: id == 1,
            hessian_index: 0,
        })) as VertexBase
    }).collect::<Vec<_>>();

    let point_vertices = problem.points.iter().map(|x| {
        id += 1;
        Rc::new(RefCell::new(PointVertex {
            id,
            params: x.clone(),
            edges: Vec::new(),
            fixed: false,
            hessian_index: 0,
        })) as VertexBase
    }).collect::<Vec<_>>();
    id = 0;
    for x in  problem.observations {
        id += 1;
        let edge = Rc::new(RefCell::new( Point3dProjectEdge {
            id,
            vertices: Vec::new(),
            sigma: na::DMatrix::<f64>::identity(2, 2),
            measurement: na::dvector![x.x, x.y],
        }
        )) as EdgeBase;
        edge.borrow_mut().add_vertex(camera_vertices[x.camera_idx].clone());
        edge.borrow_mut().add_vertex(point_vertices[x.point_idx].clone());
        graph.add_edge(&edge);
    }
    graph.add_vertex_set(camera_vertices);
    graph.add_vertex_set(point_vertices);
    println!("params: {}", graph.vertex2param().norm()); 
    graph.optimize();
    println!("params: {}", graph.vertex2param().norm()); 
    savePly("output_optimized.ply", &graph.vertex2param().as_slice()[9 * num_camera ..]);
}