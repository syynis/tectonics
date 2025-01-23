use std::f32::consts::{PI, TAU};

use vek::Vec2;

use crate::segment::Segment;

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
    pub fn new(pos: Vec2<f32>) -> Self {
        Self {
            pos,
            vel: Vec2::zero(),
            rot: 0.0,
            ang_vel: 0.0,
            mass: 0.0,
            inertia: 0.0,
            avg_height: 0.0,
            convection: 10.0,
            growth: 0.05,
            segments: Vec::new(),
        }
    }

    pub fn step(&mut self, segments: &mut Vec<Segment>, heatmap: &Vec<f64>) {
        // Update heights
        let langmuir = |k, x| -> f32 { k * x / (1.0 + k * x) };
        for seg_idx in &self.segments {
            let seg = &mut segments[*seg_idx];
            let seg_pos: Vec2<usize> = seg.pos.as_();
            let temp = heatmap[seg_pos.y * 1024 + seg_pos.x] as f32;
            let rate = self.growth * (1.0 - temp);
            let mut g = rate * (1.0 - temp - seg.density * seg.thickness);
            if (g < 0.0) {
                g *= 0.05;
            }
            let d = langmuir(3.0, 1.0 - temp);
            seg.mass += seg.area * g * d;
            seg.thickness += g;
            seg.update_bouyancy();
        }
        // Convect

        let mut acc = Vec2::zero();
        let mut torque = 0.0;

        let calc_force = move |pos| -> Vec2<f32> {
            let mut fx = 0.0;
            let mut fy = 0.0;

            Vec2::new(fx, fy)
        };

        let calc_angle = |v: Vec2<f32>| -> f32 {
            if v.x == 0.0 && v.y == 0.0 {
                return 0.0;
            }
            if v.x == 0.0 && v.y > 0.0 {
                return PI / 2.0;
            }
            if v.x == 0.0 && v.y < 0.0 {
                return 3.0 * PI / 2.0;
            }
            let mut a = TAU + (v.y / v.x).atan();
            if v.x < 0.0 {
                a += PI;
            }
            a
        };

        for s in &self.segments {
            let s = &mut segments[*s];
            let force = calc_force(s.pos);
            let dir = s.pos - self.pos;

            acc -= self.convection * force;
            torque -= self.convection
                * dir.magnitude()
                * force.magnitude()
                // * (force.angle_between(dir)).sin()
                * (calc_angle(force)-calc_angle(dir)).sin()
        }

        let dt = 0.025;
        self.vel += dt * acc / self.mass;
        self.ang_vel += dt * torque / self.inertia;
        self.pos += dt * self.vel;
        self.rot += dt * self.ang_vel;

        if self.rot > TAU {
            self.rot -= TAU;
        }
        if self.rot < 0.0 {
            self.rot += TAU;
        }

        for s in &self.segments {
            let s = &mut segments[*s];
            let dir = s.pos - (self.pos - dt * self.vel);
            // let angle = dir.angle_between(self.rot - dt * self.ang_vel);
            let angle = calc_angle(dir) - self.rot - dt * self.ang_vel;
            let var_name = self.rot + angle;
            let eff_vec = dir.magnitude() * Vec2::new(var_name.cos(), var_name.sin());
            s.vel = (self.pos + eff_vec - s.pos);
            s.pos = self.pos + eff_vec;
        }
    }
    pub fn recenter(&mut self, segments: &mut Vec<Segment>) {
        let mut new_pos = Vec2::zero();
        let mut new_inertia = 0.0;
        let mut new_mass = 0.0;
        for s in &self.segments {
            let s = &mut segments[*s];
            new_pos += s.pos;
            new_mass += s.mass;
            new_inertia += (new_pos - s.pos).magnitude_squared() * s.mass;
        }
        new_pos /= self.segments.len() as f32;
        self.pos = new_pos;
        self.inertia = new_inertia;
        self.mass = new_mass;
    }

    fn delete_oob_segments(&mut self, segments: &mut Vec<Segment>, dim: Vec2<i32>) {
        let dim: Vec2<f32> = dim.as_();
        let mut to_remove = Vec::new();
        for (s_id, s) in self.segments.iter().enumerate() {
            let s = &mut segments[*s];
            if !((0.0..=dim.x).contains(&s.pos.x) && (0.0..=dim.y).contains(&s.pos.y)) {
                s.alive = false;
                to_remove.push(s_id);
            }
        }
        for s_id in to_remove.iter().rev() {
            self.segments.remove(*s_id);
        }

        if !to_remove.is_empty() {
            self.recenter(segments);
        }
    }
}
