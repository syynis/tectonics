use std::ops::RangeInclusive;

use vek::{Clamp, Vec2};

// Points should be ordered
pub fn make_indexmap(points: &Vec<Vec2<f32>>, dim: Vec2<usize>) -> Vec<usize> {
    (0..dim.y)
        .map(|y| assign_index_row(points, y, dim))
        .flatten()
        .collect()
}

fn assign_index_row(
    points: &Vec<Vec2<f32>>,
    row: usize,
    dim: Vec2<usize>,
) -> impl Iterator<Item = usize> + '_ {
    // If points are ordered and distributed evenly this is a decent guess for the nearest point
    let mut guess =
        (((row * points.len()) as f64 / dim.x as f64) as usize).clamped(0, points.len() - 1);

    (0..dim.x).map(move |x| {
        let p = Vec2::new(x, row);
        let nearest = find_nearest(&points, p, guess);
        guess = nearest;
        nearest
    })
}

fn find_nearest2(points: &Vec<Vec2<f32>>, p: Vec2<usize>, guess: usize, dim: Vec2<usize>) -> usize {
    let p_orig = p;
    let p: Vec2<f32> = p.as_();
    let dist = p.distance(points[guess]).ceil() as usize;
    let lower_bound = p_orig.y.saturating_sub(dist);
    let upper_bound = (p_orig.y + dist).clamped(0, dim.y);

    let search_range = compute_search_range(points, lower_bound, upper_bound, dim);

    let mut current = *search_range.start();
    let mut closest = current;
    let mut closest_dist = p.distance_squared(points[current]).ceil();
    current += 1;
    while current < *search_range.end() && current < points.len() {
        let dist = p.distance_squared(points[current]);
        if dist < closest_dist {
            closest = current;
            closest_dist = dist;
        }
        current += 1;
    }
    closest
}

fn find_nearest(points: &Vec<Vec2<f32>>, p: Vec2<usize>, _guess: usize) -> usize {
    let p: Vec2<f32> = p.as_();
    points
        .iter()
        .map(|e| e.distance_squared(p))
        .enumerate()
        .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0
}

fn compute_search_range(
    points: &Vec<Vec2<f32>>,
    lower_bound: usize,
    upper_bound: usize,
    dim: Vec2<usize>,
) -> RangeInclusive<usize> {
    struct Border {
        index: usize,
        y: usize,
    }
    let lower = if points[0].y <= lower_bound as f32 {
        0
    } else {
        let mut min = Border { index: 0, y: 0 };
        let mut max = Border {
            index: points.len() - 1,
            y: dim.y,
        };
        let mut res = None;
        loop {
            let current = ((points.len() as f32 * (lower_bound - min.y) as f32
                / (max.y - min.y) as f32) as usize)
                .clamped(min.index + 1, max.index - 1);
            let current_y = points[current].y.floor() as usize;
            if current_y >= lower_bound {
                let prev_y = points[current - 1].y.floor() as usize;
                if prev_y < lower_bound {
                    res = Some(current);
                }
                min = Border {
                    index: current,
                    y: current_y,
                };
            } else {
                max = Border {
                    index: current,
                    y: current_y,
                };
            }
        }

        res.unwrap()
    };

    let upper = 10;
    lower..=upper
}
