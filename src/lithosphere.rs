use vek::*;

use crate::{plate::Plate, segment::Segment, MapSizeLg};

#[derive(Default, Clone, Copy)]
pub struct GlobalParameters {
    pub max_plate_speed: i32,
    pub subduction_distance: f64,
    pub min_altitude: u32,
    pub max_altitude: u32,
    pub base_uplift: f64,
}

pub struct Lithosphere {
    pub plates: Vec<Plate>,
    pub segments: Vec<Segment>,
    pub heatmap: Vec<f64>,
    pub index_map: Vec<usize>,
    pub iteration: u32,
    pub dimension: Vec2<i32>,
    // pub parameters: GlobalParameters,
}

impl Lithosphere {
    pub fn step(&mut self) {}

    pub fn calculate_border(&mut self) {}
}
