use hashbrown::HashSet;
use rand::distributions::Uniform;
use vek::*;

use crate::{
    grid::Grid,
    plate::{CrustKind, CrustSample, Plate, PlateArea},
    uniform_idx_as_vec2,
    util::{wrap_pos, CARDINALS},
    vec2_as_uniform_idx, MapSizeLg, Rng,
};

pub type Alt = f64;

#[derive(Default)]
pub struct GlobalParameters {
    pub max_plate_speed: u32,
    pub subduction_distance: u32,
    pub min_altitude: u32,
    pub max_altitude: u32,
    pub base_uplift: u32,
}
pub struct Lithosphere {
    pub plates: Vec<Plate>,
    pub occ_map: Vec<Option<usize>>,
    pub height: Vec<Alt>,
    pub iteration: u32,
    pub dimension: Vec2<i32>,
    pub dimension_lg: MapSizeLg,
    pub parameters: GlobalParameters,
}

impl Lithosphere {
    pub fn generate(
        map_size_lg: MapSizeLg,
        parameters: GlobalParameters,
        num_plates: usize,
        rng: &mut impl Rng,
        alt: &[Alt],
    ) -> Self {
        let plates = Self::create_plates(map_size_lg, num_plates, rng, alt);
        let mut occ_map = vec![None; map_size_lg.chunks_len()];

        for (plate_idx, plate) in plates.iter().enumerate() {
            for (pos, sample) in plate.samples.iter() {
                if sample.is_some() {
                    let pos = wrap_pos(map_size_lg.chunks().as_(), plate.origin + pos);
                    occ_map[vec2_as_uniform_idx(map_size_lg, pos)] = Some(plate_idx);
                }
            }
        }

        /*
        if occ_map.iter().all(|x| x.is_some()) {
            println!("occupancy is ok");
        }
        */

        Self {
            plates,
            occ_map,
            height: alt.to_vec(),
            iteration: 1,
            dimension: map_size_lg.chunks().as_(),
            dimension_lg: map_size_lg,
            parameters,
        }
    }

    fn create_plates(
        map_size_lg: MapSizeLg,
        num_plates: usize,
        rng: &mut impl Rng,
        alt: &[Alt],
    ) -> Vec<Plate> {
        let mut plate_area: Vec<PlateArea> = Vec::new();
        let map_sz = map_size_lg.chunks().as_::<i32>();
        let mut centers = Vec::new();

        let between = Uniform::from(0usize..map_size_lg.chunks_len());
        // Choose random points on the map
        (0..num_plates).for_each(|plate| {
            let origin = uniform_idx_as_vec2(map_size_lg, rng.sample(between));
            centers.push(origin);
            plate_area.push(PlateArea {
                border: vec![origin],
                north: origin.y,
                east: origin.x,
                south: origin.y,
                west: origin.x,
                dimension: Vec2::one(),
            });
        });

        // Grow plates
        let mut used: HashSet<Vec2<i32>> = HashSet::default();
        let mut plates = vec![Vec::new(); num_plates];
        for (idx, area) in plate_area.iter().enumerate() {
            used.insert(*area.border.first().unwrap());
        }
        while plates.iter().fold(0, |acc, x| acc + x.len()) < map_size_lg.chunks_len() {
            for (idx, area) in plate_area.iter_mut().enumerate() {
                if area.border.is_empty() {
                    continue;
                }
                let border_idx = rng.gen_range(0..area.border.len());
                let border_pos = area.border[border_idx];
                plates[idx].push(border_pos);

                let north = (border_pos.y + 1) % map_sz.y;
                let east = (border_pos.x + 1) % map_sz.x;
                let south = (border_pos.y + map_sz.y - 1) % map_sz.y;
                let west = (border_pos.x + map_sz.x - 1) % map_sz.x;

                let above = Vec2::new(border_pos.x, north);
                if used.get(&above).is_none() {
                    used.insert(above);
                    area.border.push(above);
                    if (area.north + 1) % map_sz.y == north {
                        area.north = north;
                        area.dimension.y += 1;
                    }
                };

                let right = Vec2::new(east, border_pos.y);
                if used.get(&right).is_none() {
                    used.insert(right);
                    area.border.push(right);
                    if (area.east + 1) % map_sz.x == east {
                        area.east = east;
                        area.dimension.x += 1;
                    }
                };

                let below = Vec2::new(border_pos.x, south);
                if used.get(&below).is_none() {
                    used.insert(below);
                    area.border.push(below);
                    if area.south == (south + 1) % map_sz.y {
                        area.south = south;
                        area.dimension.y += 1;
                    }
                };

                let left = Vec2::new(west, border_pos.y);
                if used.get(&left).is_none() {
                    used.insert(left);
                    area.border.push(left);
                    if area.west == (west + 1) % map_sz.x {
                        area.west = west;
                        area.dimension.x += 1;
                    }
                };

                area.border.swap_remove(border_idx);
            }
        }

        let mut res: Vec<Plate> = Vec::new();

        (0..num_plates).for_each(|plate| {
            let area = &plate_area[plate];

            let mut samples = Grid::new(area.dimension, None);
            let origin = Vec2::new(area.west, area.south);
            plates[plate].iter().for_each(|pos| {
                let sample_alt = alt[vec2_as_uniform_idx(map_size_lg, *pos)];
                let kind = if sample_alt >= 0.0 {
                    CrustKind::Continential
                } else {
                    CrustKind::Oceanic
                };
                let sample = CrustSample {
                    age: 0,
                    alt: sample_alt,
                    fold_dir: Vec2::zero(),
                    kind,
                };
                samples.set(wrap_pos(map_sz, *pos - origin), Some(sample));
            });
            let plate = Plate {
                samples,
                border: area.border.clone(),
                mass: plate as f32, // TODO hack for debugging
                com: centers[plate],
                vel: Vec2::new(rng.gen_range(-3..=3), rng.gen_range(-3..=3)),
                rot: 0.0,
                origin, // TODO probably want a float position too for more variable velocities
                dimension: area.dimension,
                world_dimension: map_sz,
            };
            res.push(plate);
        });
        res
    }

    pub fn step(&mut self) {
        println!("New simulation step: {}", self.iteration);

        let old_occ_map = self.occ_map.clone();

        for plate in &mut self.plates {
            plate.step();
        }

        // Fill gaps with oceanic crust
        // TODO instead of just checking occupancy of previous iteration, consider
        // shortest distance to plate
        for x in 0..self.dimension.x {
            for y in 0..self.dimension.y {
                let wpos = Vec2::new(x, y);

                let unoccupied = self.plates.iter().all(|plate| {
                    let rpos = plate.wpos_to_rpos(wpos);
                    if let Some(rpos) = rpos {
                        match plate.samples.get(rpos) {
                            Some(sample) => sample.is_none(),
                            None => true,
                        }
                    } else {
                        true
                    }
                });

                if unoccupied {
                    let idx = vec2_as_uniform_idx(self.dimension_lg, wpos);
                    if let Some(plate_idx) = old_occ_map[idx] {
                        self.plates[plate_idx].add_crust(wpos);
                    }
                }
            }
        }

        self.occ_map.fill(None);
        self.height.fill(0.);
        for (plate_idx, plate) in self.plates.iter().enumerate() {
            for (rpos, sample) in plate.samples.iter() {
                if let Some(sample) = sample {
                    let wpos = wrap_pos(self.dimension, plate.origin + rpos);
                    let wpos_idx = vec2_as_uniform_idx(self.dimension_lg, wpos);
                    let occupancy = self.occ_map[wpos_idx];

                    if let Some(other_plate_idx) = occupancy {
                        let other_plate = &self.plates[other_plate_idx];
                        let other_rpos = other_plate
                            .wpos_to_rpos(wpos)
                            .expect("Occupancy tells us this plate contains wpos");
                        let other_sample = other_plate.samples.get(other_rpos).expect(
                            "Occupancy tells us this plate contains rpos (derived from wpos)",
                        );

                        let relative_speed = (plate.vel - other_plate.vel).map(|x| x.abs());

                        if let Some(other_sample) = other_sample {
                            let is_oceanic = matches!(sample.kind, CrustKind::Oceanic);
                            let is_other_oceanic = matches!(other_sample.kind, CrustKind::Oceanic);

                            // we have to handle oceanic - oceanic, oceanic - continential

                            let speed_transfer = (relative_speed.magnitude_squared() as f64)
                                / (self.parameters.max_plate_speed.pow(2) as f64);
                            let speed_transfer = speed_transfer.sqrt();
                            let height_transfer = |alt: Alt| -> Alt {
                                ((alt + self.parameters.min_altitude as f64)
                                    / (self.parameters.max_altitude as f64
                                        + self.parameters.min_altitude as f64))
                                    .powi(2)
                            };

                            // TODO distance transfer

                            let uplift =
                                |alt: Alt| -> f64 { speed_transfer * height_transfer(alt) };
                            if is_oceanic {
                                if is_other_oceanic {
                                    // Oceanic - Oceanic
                                    if sample.age > other_sample.age {
                                        // We subduct
                                    } else {
                                        // Other subduct

                                        let uplift = uplift(other_sample.alt);
                                        println!("uplift {}", uplift);
                                        self.occ_map[wpos_idx] = Some(plate_idx);
                                        self.height[wpos_idx] = sample.alt + uplift;
                                    }
                                } else {
                                    // Oceanic - Continential
                                    // We subduct
                                }
                            } else {
                                if is_other_oceanic {
                                    // Other Subduct
                                    self.occ_map[wpos_idx] = Some(plate_idx);
                                    self.height[wpos_idx] = sample.alt;
                                } else {
                                    // TODO forced subduction
                                }
                            }
                        }
                    } else {
                        // Noone here yet
                        self.occ_map[wpos_idx] = Some(plate_idx);
                        self.height[wpos_idx] = sample.alt;
                    }
                }
            }
        }

        assert!(
            self.occ_map.iter().all(|x| x.is_some()),
            "everything should be occupied"
        );

        self.iteration += 1;
    }

    pub fn calculate_border(&mut self) {
        let mut borders = Vec::new();

        for (plate_idx, plate) in self.plates.iter().enumerate() {
            let mut border = Vec::new();

            for (rpos, _) in plate.samples.iter().filter(|(_, s)| s.is_some()) {
                let wpos = wrap_pos(self.dimension, plate.origin + rpos);
                let wpos_idx = vec2_as_uniform_idx(self.dimension_lg, wpos);
                let occ = self.occ_map[wpos_idx];
                let mut is_border = false;
                for neighbor in CARDINALS.iter() {
                    let npos = wrap_pos(self.dimension, wpos + neighbor);
                    let npos_idx = vec2_as_uniform_idx(self.dimension_lg, npos);
                    let n_occ = self.occ_map[npos_idx];
                    if n_occ != occ {
                        is_border = true;
                    }
                }

                if is_border {
                    border.push(rpos);
                }
            }

            borders.push(border);
        }

        for (idx, border) in borders.iter().enumerate() {
            self.plates[idx].border = border.clone();
        }
    }
}
