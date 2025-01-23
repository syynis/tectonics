use std::{
    ops::{Add, Mul},
    time::Instant,
};

use lithosphere::Lithosphere;
use minifb::*;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use plate::Plate;
use poisson::PoissonSampler;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use segment::Segment;
use vek::Vec2;
use voronoi::make_indexmap;

pub mod distance_transform;
pub mod grid;
pub mod lithosphere;
pub mod plate;
pub mod poisson;
pub mod segment;
pub mod util;
pub mod voronoi;

const ZOOM_LG: usize = 0;
const W: usize = 1024;
const H: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct MapSizeLg(pub Vec2<u32>);

impl MapSizeLg {
    pub fn new(map_size_lg: Vec2<u32>) -> Self {
        Self(map_size_lg)
    }

    pub fn vec(self) -> Vec2<u32> {
        self.0
    }

    pub fn chunks(self) -> Vec2<u16> {
        Vec2::new(1 << self.0.x, 1 << self.0.y)
    }

    pub fn chunks_len(self) -> usize {
        1 << (self.0.x + self.0.y)
    }
}

impl From<MapSizeLg> for Vec2<u32> {
    fn from(size: MapSizeLg) -> Self {
        size.vec()
    }
}

pub fn uniform_idx_as_vec2(map_size_lg: MapSizeLg, idx: usize) -> Vec2<i32> {
    let x_mask = (1 << map_size_lg.vec().x) - 1;
    Vec2::new((idx & x_mask) as i32, (idx >> map_size_lg.vec().x) as i32)
}

pub fn vec2_as_uniform_idx(map_size_lg: MapSizeLg, idx: Vec2<i32>) -> usize {
    ((idx.y as usize) << map_size_lg.vec().x) | idx.x as usize
}

fn main() {
    let mut buf = vec![0; W * H];
    let mut win = Window::new("Tectonic Plates", W, H, WindowOptions::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(0);

    let mut poisson = PoissonSampler::new(Vec2::new((W >> ZOOM_LG) as f32, (H >> ZOOM_LG) as f32));

    poisson.sample_multiple(&mut rng, None, 24.0);
    poisson.sort();

    println!("points {}", poisson.points().len(),);
    let before = Instant::now();
    let index_map = make_indexmap(poisson.points(), Vec2::new(W, H));
    let after = Instant::now();
    println!(
        "Computing index map took {} seconds",
        (after - before).as_secs_f32()
    );
    for x in 0..W {
        for y in 0..H {
            let cell = index_map[y * W + x] % 256;
            let cell = cell as u8;
            // set(Vec2::new(x as i32, y as i32), &mut buf, (cell, cell, cell));
        }
    }

    // for p in poisson.points() {
    //     set((*p).as_(), &mut buf, (50, 50, 200));
    // }

    let mut plate_points = PoissonSampler::new(Vec2::new(W as f32, H as f32));
    plate_points.sample_multiple(&mut rng, None, 64.0);

    let mut plates: Vec<Plate> = plate_points
        .points()
        .choose_multiple(&mut rng, 12)
        .map(|p| Plate::new(*p))
        .collect();

    let mut segments = Vec::new();
    for p in poisson.points() {
        let (plate_idx, plate) = plates
            .iter_mut()
            .enumerate()
            .min_by(|(_, plate), (_, plate2)| {
                plate
                    .pos
                    .distance_squared(*p)
                    .partial_cmp(&plate2.pos.distance_squared(*p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let segment = Segment::new(*p, plate_idx);
        let segment_idx = segments.len();

        segments.push(segment);
        plate.segments.push(segment_idx);
    }

    for (i, p) in plates.iter().enumerate() {
        println!("{}", p.segments.len());
        let i = (i + 1) as u8;
        let color = (i, i, i);
        // set(p.pos.as_(), &mut buf, color);

        for child in &p.segments {
            let s = &segments[*child];
            // set(s.pos.as_(), &mut buf, color);
        }
    }

    let noise: Fbm<Perlin> = Fbm::new(0)
        .set_octaves(6)
        .set_persistence(0.5)
        .set_frequency(1.0 / (((1 << 6) * 64) as f64))
        .set_lacunarity(2.0);
    let mut heatmap = vec![0.0; W * H];
    for i in 0..W {
        for j in 0..H {
            heatmap[j * W + i] = noise
                .get((Vec2::new(i as f64, j as f64) * 10.0).into_array())
                .clamp(-1.0, 1.0)
                .mul(0.5)
                .add(0.5);
        }
    }
    for i in 0..W {
        for j in 0..H {
            let heat = heatmap[j * W + i] * 255.0;
            let heat = heat as u8;
            set(Vec2::new(i as i32, j as i32), &mut buf, (heat, heat, heat))
        }
    }

    let mut lithospere = Lithosphere {
        plates,
        segments,
        heatmap,
        index_map,
        iteration: 0,
        dimension: Vec2::new(W as i32, H as i32),
    };

    while win.is_open() {
        for plate in lithospere.plates.iter_mut() {
            plate.step(&mut lithospere.segments, &lithospere.heatmap);
            plate.recenter(&mut lithospere.segments);
        }

        for i in 0..W {
            for j in 0..H {
                let idx = lithospere.index_map[j * W + i];
                let segment_height = lithospere.segments[idx].height * 255.0;
                set(
                    Vec2::new(i as i32, j as i32),
                    &mut buf,
                    (0, 0, segment_height as u8),
                );
            }
        }

        win.update_with_buffer(&buf, W, H).unwrap();
        if win.is_key_pressed(Key::Q, KeyRepeat::No) {
            break;
        }
    }
}

fn set(pos: Vec2<i32>, buf: &mut [u32], color: (u8, u8, u8)) {
    let pos = pos * (1 << ZOOM_LG);
    for zx in 0..(1 << ZOOM_LG) as i32 {
        for zy in 0..(1 << ZOOM_LG) as i32 {
            let pos = pos + Vec2::new(zx, zy);
            let idx = (H - pos.y as usize - 1) * W + pos.x as usize;
            if idx < W * H {
                buf[idx] = u32::from_le_bytes([color.0, color.1, color.2, 0]);
            }
        }
    }

    // let idx = (H - pos.y as usize - 1) * W + pos.x as usize;
    // if idx < W * H {
    //     buf[idx] = u32::from_le_bytes([color.0, color.1, color.2, 0]);
    // }
}

fn cast_u32x8_u8x32(a: [u32; 8]) -> [u8; 32] {
    let mut r = [0; 32];
    for i in 0..8 {
        let a = a[i].to_ne_bytes();
        for j in 0..4 {
            r[i * 4 + j] = a[j];
        }
    }
    r
}

pub fn diffuse(mut a: u32) -> u32 {
    a ^= a.rotate_right(23);
    a.wrapping_mul(2654435761)
}

pub fn diffuse_mult(v: &[u32]) -> u32 {
    let mut state = (1 << 31) - 1;
    for e in v {
        state = diffuse(state ^ e);
    }
    state
}

pub fn rng_state(mut x: u32) -> [u8; 32] {
    let mut r: [u32; 8] = [0; 8];
    for s in &mut r {
        x = diffuse(x);
        *s = x;
    }
    cast_u32x8_u8x32(r)
}
