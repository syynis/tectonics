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

    pub fn step(&mut self, segments: &mut Vec<Segment>, heatmap: &Vec<f64>, dim: Vec2<i32>) {
        // Update heights
        let langmuir = |k, x| -> f32 { k * x / (1.0 + k * x) };
        for seg_idx in &self.segments {
            let seg = &mut segments[*seg_idx];
            if !seg.alive {
                continue;
            }
            let seg_pos: Vec2<usize> = seg.pos.as_();
            let temp = heatmap[seg_pos.y * 1024 + seg_pos.x] as f32;
            let rate = self.growth * (1.0 - temp);
            let mut g = rate * (1.0 - temp) * (1.0 - temp - seg.density * seg.thickness);
            if g < 0.0 {
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

        let calc_force = move |pos: Vec2<i32>| -> Vec2<f32> {
            let mut fx = 0.0;
            let mut fy = 0.0;

            if pos.x > 0 && pos.x < dim.x - 1 && pos.y > 0 && pos.y < dim.y - 1 {
                fx = (heatmap[(pos.y * dim.x + pos.x + 1) as usize]
                    - heatmap[(pos.y * dim.x + pos.x - 1) as usize])
                    / 2.0;
                fy = -(heatmap[((pos.y + 1) * dim.x + pos.x) as usize]
                    - heatmap[((pos.y - 1) * dim.x + pos.x) as usize])
                    / 2.0;
            }

            Vec2::new(fx as f32, fy as f32)
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
            if !s.alive {
                continue;
            }
            let force = calc_force(s.pos.as_());
            let dir = s.pos - self.pos;

            acc -= self.convection * force;
            torque -= self.convection
                * dir.magnitude()
                * force.magnitude()
                * (calc_angle(force) - calc_angle(dir)).sin()
        }

        // let dt = 0.025;
        let dt = 0.1;
        println!("old pos {}", self.pos);
        println!("old vel {}", self.vel);
        println!("old ang_vel {}", self.ang_vel);
        println!("old rot {}", self.rot);
        println!("mass {}", self.mass);
        println!("inertia {}", self.inertia);
        self.vel += dt * acc / self.mass;
        self.ang_vel += dt * torque / self.inertia;
        self.pos += dt * self.vel;
        self.rot += dt * self.ang_vel;

        println!("new pos {}", self.pos);
        println!("new vel {}", self.vel);
        println!("new ang_vel {}", self.ang_vel);
        println!("new rot {}", self.rot);
        if self.rot > TAU {
            self.rot -= TAU;
        }
        if self.rot < 0.0 {
            self.rot += TAU;
        }

        for s in &self.segments {
            let s = &mut segments[*s];
            if !s.alive {
                continue;
            }
            let dir = s.pos - (self.pos - dt * self.vel);
            let angle = calc_angle(dir) - (self.rot - dt * self.ang_vel);
            let new_angle = self.rot + angle;
            let eff_vec = dir.magnitude() * Vec2::new(new_angle.cos(), new_angle.sin());
            s.vel = self.pos + eff_vec - s.pos;
            s.pos = self.pos + eff_vec;
        }
    }
    pub fn recenter(&mut self, segments: &mut Vec<Segment>) {
        println!("recenter");
        let mut new_pos = Vec2::zero();
        let mut new_inertia = 0.0;
        let mut new_mass = 0.0;
        let mut num_segments = 0;
        for s in &self.segments {
            let s = &mut segments[*s];
            if !s.alive {
                continue;
            }
            new_pos += s.pos;
            new_mass += s.mass;
            new_inertia += (new_pos - s.pos).magnitude_squared() * s.mass;
            num_segments += 1;
        }
        new_pos /= num_segments as f32;
        self.pos = new_pos;
        self.inertia = new_inertia;
        self.mass = new_mass;
        println!("pos {new_pos}, mass {new_mass}, inertia {new_inertia}");
    }

    pub fn delete_oob_segments(&mut self, segments: &mut Vec<Segment>, dim: Vec2<i32>) {
        let dim: Vec2<f32> = dim.as_();
        let mut any_deleted = false;
        for s in &self.segments {
            let s = &mut segments[*s];
            if !((0.0..dim.x).contains(&s.pos.x) && (0.0..dim.y).contains(&s.pos.y)) {
                s.alive = false;
                any_deleted = true;
            }
        }
        self.segments.retain(|s| segments[*s].alive);

        if any_deleted {
            self.recenter(segments);
        }
    }
}
