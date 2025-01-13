use std::f32::consts::{PI, SQRT_2, TAU};

use rand::{seq::SliceRandom, Rng};
use vek::{Aabr, Vec2};
pub struct PoissonSampler {
    points: Vec<Vec2<f32>>,
    dimension: Vec2<f32>,
}
impl PoissonSampler {
    pub fn new(dimension: Vec2<f32>) -> Self {
        Self {
            points: Vec::new(),
            dimension,
        }
    }

    pub fn sample_single(
        &mut self,
        rng: &mut impl Rng,
        radius: f32,
        near: Vec2<f32>,
        area: Option<Aabr<f32>>,
        attempts: u64,
    ) -> Option<Vec2<f32>> {
        // No points
        if self.points.is_empty() {
            self.points.push(near);
            return Some(near);
        }
        // Cell size
        let length = radius / SQRT_2;
        // Sample area
        let area = area.unwrap_or(Aabr {
            min: Vec2::broadcast(0.0),
            max: self.dimension,
        });
        let (min, max) = (area.min, area.max);
        let size = max - min;
        // Sample area scaled by cell size
        let size_scaled: Vec2<i32> = (size / length).ceil().as_();
        // Init grid
        let mut grid = vec![None; size_scaled.x as usize * size_scaled.y as usize];

        let insert_grid = |grid: &mut Vec<Option<usize>>, p: Vec2<f32>, v: usize| {
            let grid_idx: Vec2<i32> = size_scaled * ((p - min) / size).as_();
            grid[grid_idx.x as usize * size_scaled.y as usize + grid_idx.y as usize] = Some(v);
        };

        // Which cells occupied by existing points already
        for (p_idx, p) in self.points.iter().enumerate() {
            insert_grid(&mut grid, *p, p_idx);
        }

        // Closest existing point near our preferred point
        let target = *self
            .points
            .iter()
            .map(|p| (p, p.distance_squared(near)))
            .min_by(|(_, dist), (_, dist2)| {
                dist.partial_cmp(dist2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
            .0;

        let mut res = None;
        // Attempt to find cell next to target which is not occupied
        for _ in 0..attempts {
            let nr = rng.gen_range(radius..2.0 * radius);
            let nt = rng.gen_range(0.0..TAU);

            let npos = target + nr * Vec2::new(nt.cos(), nt.sin());
            let ind: Vec2<i32> = size_scaled * ((npos - min) / size).as_();

            let mut free = true;

            for i in ind.x - 2..=ind.x + 2 {
                for j in ind.y - 2..=ind.y + 2 {
                    if i < 0 || i >= size_scaled.x || j < 0 || j >= size_scaled.y {
                        continue;
                    }
                    if let Some(e) = grid[(i * size_scaled.y + j) as usize] {
                        if self.points[e].distance_squared(npos) < radius.powi(2) {
                            free = false;
                            break;
                        }
                    }
                }
                if !free {
                    break;
                }
            }

            if free {
                self.points.push(npos);
                insert_grid(&mut grid, npos, self.points.len() - 1);
                res = Some(npos);
                break;
            }
        }
        return res;
    }

    pub fn sample_multiple(&mut self, rng: &mut impl Rng, area: Option<Aabr<f32>>, k: u64) {
        let mut created = 0;
        // No points
        // Sample area
        let area = area.unwrap_or(Aabr {
            min: Vec2::broadcast(0.0),
            max: self.dimension,
        });
        let (min, max) = (area.min, area.max);
        let size = max - min;
        let radius = (size.magnitude_squared() / PI / k as f32).sqrt();
        // Cell size
        let length = radius / SQRT_2;
        // Sample area scaled by cell size
        let size_scaled: Vec2<i32> = (size / length).ceil().as_();
        // Init grid
        let mut grid = vec![None; size_scaled.x as usize * size_scaled.y as usize];
        let insert_grid = |grid: &mut Vec<Option<usize>>, p: Vec2<f32>, v: usize| {
            let grid_idx: Vec2<i32> = size_scaled * ((p - min) / size).as_();
            grid[grid_idx.x as usize * size_scaled.y as usize + grid_idx.y as usize] = Some(v);
        };

        if self.points.is_empty() {
            self.points.push(area.center());
            insert_grid(&mut grid, area.center(), 0);
            created += 1;
        } else {
            // Which cells occupied by existing points already
            for (p_idx, p) in self.points.iter().enumerate() {
                insert_grid(&mut grid, *p, p_idx);
            }
        }

        let mut tries = 0;
        // Attempt to find cell next to target which is not occupied
        while created < k {
            if tries > 2 * k {
                break;
            }
            tries += 1;
            let nr = rng.gen_range(radius..2.0 * radius);
            let nt = rng.gen_range(0.0..TAU);

            let npos =
                self.points.choose(rng).unwrap().clone() + nr * Vec2::new(nt.cos(), nt.sin());
            let ind: Vec2<i32> = size_scaled * ((npos - min) / size).as_();

            let mut free = true;

            for i in ind.x - 2..=ind.x + 2 {
                for j in ind.y - 2..=ind.y + 2 {
                    if i < 0 || i >= size_scaled.x || j < 0 || j >= size_scaled.y {
                        continue;
                    }
                    if let Some(e) = grid[(i * size_scaled.y + j) as usize] {
                        if self.points[e].distance_squared(npos) < radius.powi(2) {
                            free = false;
                            break;
                        }
                    }
                }
                if !free {
                    break;
                }
            }

            if free {
                self.points.push(npos);
                insert_grid(&mut grid, npos, self.points.len() - 1);
                created += 1;
            }
        }
    }

    pub fn points(&self) -> &Vec<Vec2<f32>> {
        &self.points
    }
}
