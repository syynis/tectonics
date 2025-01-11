use minifb::*;
use vek::Vec2;

pub mod distance_transform;
pub mod grid;
pub mod lithosphere;
pub mod plate;
pub mod poisson;
pub mod segment;
pub mod util;
pub mod voronoi;

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

    while win.is_open() {
        win.update_with_buffer(&buf, W, H).unwrap();
    }
}

fn set(pos: Vec2<i32>, buf: &mut [u32], color: (u8, u8, u8)) {
    let idx = (H - pos.y as usize - 1) * W + pos.x as usize;
    if idx < W * H {
        buf[idx] = u32::from_le_bytes([color.0, color.1, color.2, 0]);
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
