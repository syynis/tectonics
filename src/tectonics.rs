use hashbrown::HashSet;
use rand::distributions::Uniform;
use vek::*;

use crate::{grid::Grid, uniform_idx_as_vec2, vec2_as_uniform_idx, MapSizeLg, Rng};

pub type Alt = f64;

#[derive(Clone, Debug)]
pub enum CrustKind {
    Continential,
    Oceanic,
}

#[derive(Clone, Debug)]
pub struct CrustSample {
    pub age: u32,
    pub alt: Alt,
    pub fold_dir: Vec2<i32>,
    pub kind: CrustKind,
}

#[derive(Default)]
pub struct PlateArea {
    pub border: Vec<Vec2<i32>>,
    pub north: i32,
    pub east: i32,
    pub south: i32,
    pub west: i32,
    pub dimension: Vec2<i32>,
}

pub struct Plate {
    pub samples: Grid<Option<CrustSample>>,
    pub border: Vec<Vec2<i32>>,
    mass: f32,
    pub com: Vec2<i32>,
    pub vel: Vec2<i32>,
    rot: f32,
    // TODO think about consolidating this into aabr
    pub origin: Vec2<i32>,
    pub dimension: Vec2<i32>,
    pub world_dimension: Vec2<i32>,
}

impl Plate {
    pub fn step(&mut self) {
        for (pos, sample) in self.samples.iter_mut() {
            if let Some(sample) = sample {
                sample.age += 1;
            }
        }
        self.com = wrap_pos(self.world_dimension, self.com + self.vel);
        self.origin = wrap_pos(self.world_dimension, self.origin + self.vel);
    }

    pub fn grow_to_include(&mut self, wpos: Vec2<i32>) {
        let dist_min = self.origin - wpos;
        let dist_max = wpos - (self.origin + self.dimension - 1);

        // Shift max if smaller than min
        let dist_max = dist_max
            + wpos.map3(
                self.origin,
                self.world_dimension,
                |w, o, d| if w < o { d } else { 0 },
            );

        // Calculate distance if wrapping around edge
        let dist_min_wrapped = wrap_pos(self.world_dimension, dist_min);
        let dist_max_wrapped = wrap_pos(self.world_dimension, dist_max);

        let abs_min = |a: Vec2<i32>, b: Vec2<i32>| -> Vec2<i32> {
            a.map2(b, |a, b| if a.abs() < b.abs() { a } else { b })
        };

        // Take the smaller absolute distance of the normal or wrapped distance
        let dist_min = abs_min(dist_min, dist_min_wrapped);
        let dist_max = abs_min(dist_max, dist_max_wrapped);

        // TODO Try to make this more concise
        // Take the smaller distance on horizontal and vertical seperately
        let delta3 = abs_min(dist_min, dist_max);
        // Set the larger axis to zero
        let dist_min = dist_min.map2(delta3, |a, b| if a == b { a } else { 0 });
        let dist_max = dist_max.map2(delta3, |a, b| if a == b { a } else { 0 });

        // Clamp to to positive values
        let dist_min: Vec2<i32> = Vec2::max(dist_min, Vec2::zero());
        let dist_max: Vec2<i32> = Vec2::max(dist_max, Vec2::zero());

        // This is how much we have to grow
        // TODO Stop plate from overlapping with itself
        // TODO Stop plate from growing larger than size of the world
        let expand = dist_min + dist_max;
        let old_dimension = self.dimension;
        self.dimension += expand;

        // Shift origin
        let old_origin = self.origin;
        let new_origin = self.origin - dist_min;
        self.origin = wrap_pos(self.world_dimension, new_origin);

        // Store the difference when copying over the old grid into the new grid
        let origin_delta = old_origin - new_origin;

        // Copy over to new grid
        let mut res: Grid<Option<CrustSample>> = Grid::new(self.dimension, None);

        for y in 0..old_dimension.y {
            for x in 0..old_dimension.x {
                let pos = Vec2::new(x, y) + origin_delta;
                // TODO should be possible to move values from smaller grid instead of cloning
                res.set(
                    pos,
                    self.samples.get(Vec2::new(x, y)).and_then(|s| s.clone()),
                );
            }
        }

        self.samples = res;
    }

    // Invariant pos is in bounds
    pub fn contains(&self, wpos: Vec2<i32>) -> bool {
        let min = self.origin;
        // TODO is the wrap_pos needed here?
        let max = wrap_pos(self.world_dimension, self.origin + self.dimension);
        let max =
            max + self.world_dimension * Vec2::new((max.x < min.x) as i32, (max.y < min.y) as i32);

        let aabr_contains = |aabr: Aabr<i32>, pos: Vec2<i32>| -> (bool, bool) {
            (
                (aabr.min.x..aabr.max.x).contains(&pos.x),
                (aabr.min.y..aabr.max.y).contains(&pos.y),
            )
        };

        let aabr = Aabr { min, max };
        let (x, y) = aabr_contains(aabr, wpos);
        let (x_shifted, y_shifted) = aabr_contains(aabr, wpos + self.world_dimension);

        (x || x_shifted) && (y || y_shifted)
    }

    pub fn wpos_to_rpos(&self, wpos: Vec2<i32>) -> Option<Vec2<i32>> {
        if self.contains(wpos) {
            let rpos = wpos.map3(self.origin, self.world_dimension, |pos, origin, dim| {
                if pos < origin {
                    pos + dim
                } else {
                    pos
                }
            });

            assert!(
                (rpos - self.origin).iter().all(|x| *x >= 0),
                "wpos is in plate boundary so rpos must be valid i.e. positive"
            );

            Some(rpos - self.origin)
        } else {
            None
        }
    }

    pub fn add_crust(&mut self, wpos: Vec2<i32>) {
        // Just to be safe wrap world pos
        let wpos = wrap_pos(self.world_dimension, wpos);

        // TODO replace this by parameter
        let sample = CrustSample {
            age: 0,
            alt: 0.0,
            fold_dir: Vec2::zero(),
            kind: CrustKind::Oceanic,
        };

        match self.wpos_to_rpos(wpos) {
            Some(rpos) => match self.samples.get(rpos) {
                Some(x) => match x {
                    Some(_) => {}
                    None => {
                        self.samples.set(rpos, Some(sample));
                    }
                },
                None => {
                    unreachable!("This should not happen");
                }
            },
            None => {
                self.grow_to_include(wpos);
                let rpos = self.wpos_to_rpos(wpos);
                let rpos = rpos.expect("Should be in plate after growing");
                let res = self.samples.set(rpos, Some(sample));
            }
        }
    }
}

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
                                        //println!("uplift {}", uplift);
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
        for plate in self.plates.iter_mut() {}
    }
}

pub fn wrap_pos(dim: Vec2<i32>, pos: Vec2<i32>) -> Vec2<i32> {
    Vec2::new(wrap(dim.x, pos.x), wrap(dim.y, pos.y))
}

fn wrap(dim: i32, pos: i32) -> i32 {
    if pos < 0 {
        dim + pos
    } else {
        pos % dim
    }
}
