use vek::Vec2;

pub struct Segment {
    pub pos: Vec2<f32>,
    pub area: f32,
    pub vel: Vec2<f32>,
    pub mass: f32,
    pub thickness: f32,
    pub density: f32,
    pub height: f32,
    growth: f32,

    parent: usize,

    pub alive: bool,
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
