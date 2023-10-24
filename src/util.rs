use vek::Vec2;

pub fn wrap_pos(dim: Vec2<i32>, pos: Vec2<i32>) -> Vec2<i32> {
    Vec2::new(wrap(dim.x, pos.x), wrap(dim.y, pos.y))
}

fn wrap(dim: i32, pos: i32) -> i32 {
    if pos < 0 {
        dim + pos
    } else {
        pos % dim
    }
}
