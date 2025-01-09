use hashbrown::HashSet;
use rand::distributions::Uniform;
use vek::*;

use crate::{
    grid::Grid,
    plate::{CrustKind, CrustSample, Plate, PlateArea},
    uniform_idx_as_vec2,
    util::wrap_pos,
    vec2_as_uniform_idx, MapSizeLg, Rng,
};

pub type Alt = f64;

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
        let plates = Self::create_plates(map_size_lg, num_plates, rng, alt, parameters);
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
        parameters: GlobalParameters,
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
                border: HashSet::new(),
                border_dist: Grid::new(Vec2::new(0, 0), 0.),
                mass: plate as f32, // TODO hack for debugging
                com: centers[plate],
                vel: Vec2::new(
                    rng.gen_range(-parameters.max_plate_speed..=parameters.max_plate_speed),
                    rng.gen_range(-parameters.max_plate_speed..=parameters.max_plate_speed),
                ),
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
            plate.compute_border();
            plate.compute_distance_transform();
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

        let speed_transfer = |p1: &Plate, p2: &Plate| -> f64 {
            let relative_speed = (p1.vel - p2.vel).map(|x| x.abs());
            ((relative_speed.magnitude_squared() as f64)
                / (self.parameters.max_plate_speed.pow(2) as f64))
                .clamp(0., 1.)
        };
        let speed_transfer = |p1: &Plate, p2: &Plate| -> f64 { speed_transfer(p1, p2).sqrt() };
        let height_transfer = |alt: Alt| -> Alt {
            ((alt + self.parameters.min_altitude as f64)
                / (self.parameters.max_altitude as f64 + self.parameters.min_altitude as f64))
                .powi(2)
                .clamp(0., 1.)
        };
        let distance_transfer = |dt: &Grid<f64>, pos: Vec2<i32>| -> f64 {
            let dist_sq = dt
                .get(pos)
                .unwrap_or(&0.0)
                .clamp(0., self.parameters.subduction_distance.powi(2));
            1. - dist_sq / self.parameters.subduction_distance.powi(2)
        };
        // TODO distance transfer
        let uplift = |alt: Alt, p1: &Plate, p2: &Plate, dt: &Grid<f64>, pos: Vec2<i32>| -> f64 {
            let speed = speed_transfer(p1, p2);
            let height = height_transfer(alt);
            let dist = distance_transfer(dt, pos);
            let uplift = self.parameters.base_uplift * speed * height * dist;
            println!(
                "speed {:.3},\t height {:.3},\t dist {:.3},\t uplift {:.6}\t",
                speed, height, dist, uplift
            );
            uplift
        };

        for plate_idx in 0..self.plates.len() {
            let plate = &self.plates[plate_idx];
            for other_plate_idx in (plate_idx + 1)..self.plates.len() {
                let other_plate = &self.plates[other_plate_idx];
                let intersection = plate.aabr().intersection(other_plate.aabr()).made_valid();
                let intersection = plate.aabr().intersection(intersection);
                println!("p {}, po {}", plate_idx, other_plate_idx);
                println!(
                    "origin {}, other origin {}",
                    plate.origin, other_plate.origin
                );
                println!(
                    "dim {}, other_dim {}",
                    plate.dimension, other_plate.dimension
                );
                println!("intersection {:?}", intersection);
                for x in intersection.min.x..intersection.max.x {
                    for y in intersection.min.y..intersection.max.y {
                        let rpos = plate.origin + Vec2::new(x, y);
                        let other_rpos = other_plate.origin + Vec2::new(x, y);
                        let sample = plate.samples.get(rpos).expect("Test");
                        let other_sample = other_plate.samples.get(other_rpos).expect("Test Other");
                    }
                }
            }
        }
        for (plate_idx, plate) in self.plates.iter().enumerate() {
            let dt = &plate.border_dist;
            for (rpos, sample) in plate.samples.iter() {
                let Some(sample) = sample else {
                    continue;
                };
                let wpos = wrap_pos(self.dimension, plate.origin + rpos);
                let wpos_idx = vec2_as_uniform_idx(self.dimension_lg, wpos);
                let occupancy = self.occ_map[wpos_idx];

                if let Some(other_plate_idx) = occupancy {
                    let other_plate = &self.plates[other_plate_idx];
                    let other_dt = &other_plate.border_dist;
                    let other_rpos = other_plate
                        .wpos_to_rpos(wpos)
                        .expect("Occupancy tells us this plate contains wpos");
                    let other_sample = other_plate
                        .samples
                        .get(other_rpos)
                        .expect("Occupancy tells us this plate contains rpos (derived from wpos)");

                    let Some(other_sample) = other_sample else {
                        continue;
                    };
                    let is_oceanic = matches!(sample.kind, CrustKind::Oceanic);
                    let is_other_oceanic = matches!(other_sample.kind, CrustKind::Oceanic);

                    // we have to handle oceanic - oceanic, oceanic - continential
                    let uplift = uplift(other_sample.alt, &plate, &other_plate, dt, rpos);
                    if is_oceanic {
                        if is_other_oceanic {
                            // Oceanic - Oceanic
                            if sample.age > other_sample.age {
                                // We subduct
                            } else {
                                // Other subduct
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
                            self.height[wpos_idx] = sample.alt + uplift;
                        } else {
                            // TODO forced subduction
                        }
                    }
                } else {
                    // Noone here yet
                    self.occ_map[wpos_idx] = Some(plate_idx);
                    self.height[wpos_idx] = sample.alt;
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
        for plate in self.plates.iter_mut() {
            plate.compute_border();
        }
    }
}
