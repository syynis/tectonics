use vek::Vec2;

pub struct Plate {
    pub pos: Vec2<f32>,
    pub vel: Vec2<f32>,
    pub rot: f32,
    pub ang_vel: f32,
    pub mass: f32,
    pub inertia: f32,
    pub avg_height: f32,
    pub convection: f32,
    pub growth: f32,

    // Segment ids
    pub segments: Vec<usize>,
}

impl Plate {
    pub fn step(&mut self) {}
    pub fn recenter(&mut self) {}
}
