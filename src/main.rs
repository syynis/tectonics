use minifb::{Key, KeyRepeat, Window, WindowOptions};
use noise::{
    utils::{NoiseMapBuilder, PlaneMapBuilder},
    Fbm, Perlin,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use vek::*;

use crate::tectonics::{wrap_pos, CrustKind, CrustSample, GlobalParameters, Lithosphere};

pub mod grid;
pub mod tectonics;

const SIZE_LG: usize = 10;
const ZOOM_LG: usize = 0;
const SIZE_ZOOM_LG: usize = SIZE_LG - ZOOM_LG;
const SIZE: usize = 1 << (SIZE_LG - ZOOM_LG);
const SIZE_WINDOW: usize = 1 << SIZE_LG;
const W: usize = SIZE_WINDOW;
const H: usize = SIZE_WINDOW;
const NUM_PLATES: usize = 16;

const NEIGHS: [Vec2<i32>; 9] = {
    [
        Vec2::new(0, 0),
        Vec2::new(0, 1),
        Vec2::new(1, 1),
        Vec2::new(1, 0),
        Vec2::new(1, -1),
        Vec2::new(0, -1),
        Vec2::new(-1, -1),
        Vec2::new(-1, 0),
        Vec2::new(-1, 1),
    ]
};

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
    let map_size_lg = MapSizeLg::new(Vec2::new(SIZE_ZOOM_LG as u32, SIZE_ZOOM_LG as u32));
    let seed = 1337;
    let max_height = 2048.;

    let mut rng = ChaChaRng::from_seed(rng_state(seed));

    let mut buf = vec![0; W * H];
    let mut win = Window::new("Tectonic Plates", W, H, WindowOptions::default()).unwrap();
    let fbm = Fbm::<Perlin>::new(0);

    let plane = PlaneMapBuilder::<_, 2>::new(&fbm)
        .set_size(W, H)
        .set_x_bounds(-1.0, 1.0)
        .set_y_bounds(-1.0, 1.0)
        .build();
    let mut alt = vec![0.; W * H];
    (0..W).for_each(|x| {
        (0..H).for_each(|y| {
            alt[x + y * W] = plane.get_value(x, y).clamp(-1., 1.);
        })
    });

    /*
    println!(
        "max {}",
        alt.iter()
            .reduce(|acc, e| if acc > e { acc } else { e })
            .unwrap()
    );
    println!(
        "min {}",
        alt.iter()
            .reduce(|acc, e| if acc < e { acc } else { e })
            .unwrap()
    );
    */

    let params = GlobalParameters {
        max_plate_speed: 10,     // TODO value
        subduction_distance: 50, // TODO value
        min_altitude: max_height as u32,
        max_altitude: max_height as u32,
        base_uplift: 0, // TODO value
    };
    let mut lithosphere = Lithosphere::generate(map_size_lg, params, NUM_PLATES, &mut rng, &alt);

    let mut redraw = true;
    let mut current_plate: usize = 14.min(NUM_PLATES - 1);
    let mut view_single_plate = false;
    let mut render_dimension_border = true;
    let alt_factor = 255. / 2.;
    let steps_before_render = -1;

    for _ in 0..=steps_before_render {
        lithosphere.step();
    }

    while win.is_open() {
        if redraw {
            redraw = false;

            buf = vec![0; W * H];

            let sample_color = |sample: &CrustSample| -> (u8, u8, u8) {
                let alt = sample.alt / max_height;
                let alt = (alt_factor * (alt + 1.)) as u8;
                let color = match sample.kind {
                    CrustKind::Continential => (0, alt, 0),
                    CrustKind::Oceanic => (alt, 0, 0),
                };

                color
            };

            for idx in 0..(W * H) {
                let plate_idx = lithosphere.occ_map[idx].expect("everything should be occupied");
                if view_single_plate && plate_idx != current_plate {
                    continue;
                }

                let plate = &lithosphere.plates[plate_idx];
                let wpos = uniform_idx_as_vec2(map_size_lg, idx);
                let rpos = plate
                    .wpos_to_rpos(wpos)
                    .expect("plate should contain wpos because occupancy says so");
                let sample = plate
                    .samples
                    .get(rpos)
                    .expect("should be valid rpos")
                    .as_ref()
                    .expect("rpos should have crust sample");
                let color = sample_color(sample);
                let pos = wrap_pos(lithosphere.dimension, plate.origin + rpos);
                set(pos, &mut buf, color);
            }
            if view_single_plate && render_dimension_border {
                let plate = &lithosphere.plates[current_plate];
                let origin = plate.origin;
                let dimension = plate.dimension;
                let end = origin + dimension;

                for x in origin.x..end.x {
                    let pos = wrap_pos(lithosphere.dimension, Vec2::new(x, origin.y));
                    set(pos, &mut buf, (100, 100, 100));
                    let pos = wrap_pos(lithosphere.dimension, Vec2::new(x, end.y));
                    set(pos, &mut buf, (100, 100, 100));
                }

                for y in origin.y..end.y {
                    let pos = wrap_pos(lithosphere.dimension, Vec2::new(origin.x, y));
                    set(pos, &mut buf, (100, 100, 100));
                    let pos = wrap_pos(lithosphere.dimension, Vec2::new(end.x, y));
                    set(pos, &mut buf, (100, 100, 100));
                }
            }

            /*
            for (idx, plate) in lithosphere.plates.iter().enumerate() {
                if view_single_plate && idx != current_plate {
                    continue;
                }
                for (pos, sample) in plate.samples.iter() {
                    if let Some(sample) = sample {
                        let color = sample_color(sample);
                        let pos = wrap_pos(lithosphere.dimension, plate.origin + pos);
                        set(pos, &mut buf, color);
                    }
                }

            }
            */
        }

        if win.is_key_pressed(Key::P, KeyRepeat::No) {
            if view_single_plate {
                let plate = &lithosphere.plates[current_plate];
                println!(
                    "origin: {}, size: {}, end {}, vel: {}",
                    plate.origin,
                    plate.dimension,
                    wrap_pos(lithosphere.dimension, plate.origin + plate.dimension),
                    plate.vel
                );
            }
        }
        if win.is_key_pressed(Key::C, KeyRepeat::No) {
            current_plate = (current_plate + 1) % NUM_PLATES;
            println!("view plate {}", current_plate);
            redraw = true;
        }

        if win.is_key_pressed(Key::X, KeyRepeat::No) {
            current_plate = (current_plate as i32 - 1).wrapped(NUM_PLATES as i32) as usize;
            println!("view plate {}", current_plate);
            redraw = true;
        }

        if win.is_key_pressed(Key::T, KeyRepeat::No) {
            view_single_plate = !view_single_plate;
            buf = vec![0; W * H];
            redraw = true;
        }

        if win.is_key_pressed(Key::B, KeyRepeat::No) {
            render_dimension_border = !render_dimension_border;
            redraw = true;
        }

        if win.is_key_pressed(Key::F, KeyRepeat::No) {
            if view_single_plate {
                let plate = &mut lithosphere.plates[current_plate];
                plate.add_crust(plate.origin - 1);
                redraw = true;
            }
        }

        if win.is_key_pressed(Key::H, KeyRepeat::No) {
            if view_single_plate {
                let plate = &mut lithosphere.plates[current_plate];
                plate.add_crust(plate.origin + Vec2::new(plate.dimension.x, 0));
                redraw = true;
            }
        }

        if win.is_key_pressed(Key::G, KeyRepeat::No) {
            if view_single_plate {
                let plate = &mut lithosphere.plates[current_plate];
                plate.add_crust(plate.origin + plate.dimension);
                redraw = true;
            }
        }

        if win.is_key_pressed(Key::S, KeyRepeat::No) {
            lithosphere.step();
            redraw = true;
        }

        if win.is_key_pressed(Key::D, KeyRepeat::No) {
            if view_single_plate {
                let plate = &mut lithosphere.plates[current_plate];
                plate.step();
                redraw = true;
            }
        }

        win.update_with_buffer(&buf, W, H).unwrap();
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
