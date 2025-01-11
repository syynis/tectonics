use vek::Vec2;

pub struct Segment {
    pos: Vec2<f32>,
    area: f32,
    vel: Vec2<f32>,
    mass: f32,
    thickness: f32,
    density: f32,
    height: f32,
    growth: f32,

    parent: usize,

    alive: bool,
    colliding: bool,
}

impl Segment {
    pub fn new(pos: Vec2<f32>, parent: usize) -> Self {
        Self {
            pos,
            area: 1.0,
            vel: Vec2::broadcast(0.0),
            mass: 0.1,
            thickness: 0.1,
            density: 1.0, // mass/(area*thickness)
            height: 0.0,
            growth: 0.0,
            parent,
            alive: true,
            colliding: false,
        }
    }

    pub fn update_bouyancy(&mut self) {
        self.density = self.mass / (self.area * self.thickness);
        self.height = self.thickness * (1.0 - self.density);
    }
}
