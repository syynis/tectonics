use vek::*;

use crate::{plate::Plate, segment::Segment, voronoi::make_indexmap};

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
    pub fn init(&mut self) {
        self.update_mass();
        for p in &mut self.plates {
            p.recenter(&mut self.segments);
        }
    }
    pub fn step(&mut self) {
        self.update_mass();
        for plate in self.plates.iter_mut() {
            plate.step(&mut self.segments, &self.heatmap, self.dimension);
            plate.recenter(&mut self.segments);
        }
        for plate in &mut self.plates {
            plate.delete_oob_segments(&mut self.segments, self.dimension);
        }

        self.plates.retain(|p| !p.segments.is_empty());
        self.update_indexmap();
    }

    pub fn calculate_border(&mut self) {}

    pub fn update_indexmap(&mut self) {
        let new = make_indexmap(
            &self
                .segments
                .iter()
                .filter_map(|s| s.alive.then_some(s.pos))
                .collect(),
            self.dimension.as_(),
        );
        self.index_map = new;
    }

    pub fn update_mass(&mut self) {
        for seg in &mut self.segments {
            seg.area = 0.0;
        }
        for seg_idx in &self.index_map {
            self.segments[*seg_idx].area += 1.0;
        }
        for seg in &mut self.segments {
            seg.update_bouyancy();
        }
    }
}
