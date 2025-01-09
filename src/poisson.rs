use std::f32::consts::{SQRT_2, TAU};

use rand::Rng;
use vek::{Aabr, Vec2};
struct PoissonSampler {
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

        // Which cells occupied by existing points already
        for (p_idx, p) in self.points.iter().enumerate() {
            let grid_idx: Vec2<i32> = size_scaled * ((*p - min) / size).as_();
            grid[grid_idx.x as usize * size_scaled.y as usize + grid_idx.y as usize] = Some(p_idx);
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
                let grid_idx: Vec2<i32> = size_scaled * ((npos - min) / size).as_();
                grid[grid_idx.x as usize * size_scaled.y as usize + grid_idx.y as usize] =
                    Some(self.points.len());
                res = Some(npos);
                break;
            }
        }
        return res;
    }
}
