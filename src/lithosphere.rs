use vek::*;

use crate::{plate::Plate, MapSizeLg};

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
    pub occ_map: Vec<Option<usize>>,
    pub height: Vec<f32>,
    pub iteration: u32,
    pub dimension: Vec2<i32>,
    pub dimension_lg: MapSizeLg,
    pub parameters: GlobalParameters,
}

impl Lithosphere {
    pub fn step(&mut self) {}

    pub fn calculate_border(&mut self) {}
}
