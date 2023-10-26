use hashbrown::HashSet;
use vek::{Aabr, Vec2};

use crate::{grid::Grid, lithosphere::Alt, util::wrap_pos};

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
    pub border: HashSet<Vec2<i32>>,
    pub mass: f32,
    pub com: Vec2<i32>,
    pub vel: Vec2<i32>,
    pub rot: f32,
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
