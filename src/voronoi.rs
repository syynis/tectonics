use std::ops::RangeInclusive;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use vek::{Clamp, Vec2};

// Points should be ordered
pub fn make_indexmap(points: &Vec<Vec2<f32>>, dim: Vec2<usize>) -> Vec<usize> {
    (0..dim.y)
        .into_par_iter()
        .map(|y| assign_index_row(points, y, dim))
        .flatten()
        .collect()
}

fn assign_index_row(points: &Vec<Vec2<f32>>, row: usize, dim: Vec2<usize>) -> Vec<usize> {
    // If points are ordered and distributed evenly this is a decent guess for the nearest point
    let mut guess =
        (((row * points.len()) as f64 / dim.y as f64) as usize).clamped(0, points.len() - 1);

    (0..dim.x)
        .map(|x| {
            let p = Vec2::new(x, row);
            let nearest = find_nearest(&points, p);
            // let nearest = find_nearest2(&points, p, guess, dim);
            guess = nearest;
            nearest
        })
        .collect()
}

fn find_nearest2(points: &Vec<Vec2<f32>>, p: Vec2<usize>, guess: usize, dim: Vec2<usize>) -> usize {
    let p_orig = p;
    let p: Vec2<f32> = p.as_();
    let dist = p.distance(points[guess]).ceil() as usize;
    let lower_bound = p_orig.y.saturating_sub(dist);
    let upper_bound = (p_orig.y + dist).clamped(0, dim.y);

    println!("{p}, {dist}, {lower_bound}, {upper_bound}");

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

fn find_nearest(points: &Vec<Vec2<f32>>, p: Vec2<usize>) -> usize {
    let p: Vec2<f32> = p.as_();
    points
        .iter()
        .map(|e| e.distance_squared(p))
        .enumerate()
        .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |e| e.0)
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
    let first = 0;
    let last = points.len() - 1;
    let lower = if points[first].y >= lower_bound as f32 {
        0
    } else {
        let mut min = Border { index: first, y: 0 };
        let mut max = Border {
            index: last,
            y: dim.y,
        };
        assert!(lower_bound > min.y);
        assert!(lower_bound < max.y);

        let mut res = None;
        loop {
            let current = ((points.len() as f32 * (lower_bound - min.y) as f32
                / (max.y - min.y) as f32) as usize)
                .clamped(min.index + 1, max.index - 1);
            assert!(current >= first);
            assert!(current <= last);
            let current_y = points[current].y.floor() as usize;
            if current_y >= lower_bound {
                let prev_y = points[current - 1].y.floor() as usize;
                if prev_y < lower_bound {
                    res = Some(current);
                    break;
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

    let minimum = points[lower].y.floor() as usize;
    assert!(upper_bound >= minimum);
    let upper = if points[last].y <= upper_bound as f32 {
        last
    } else {
        let mut min = Border {
            index: first,
            y: minimum,
        };
        let mut max = Border {
            index: last,
            y: dim.y,
        };
        assert!(upper_bound >= min.y);
        assert!(upper_bound <= max.y);

        let mut res = None;
        loop {
            let current = ((points.len() as f32 * (upper_bound - min.y) as f32
                / (max.y - min.y) as f32) as usize)
                .clamped(min.index + 1, max.index - 1);
            assert!(current >= first);
            assert!(current <= last);
            let current_y = points[current].y.floor() as usize;
            if current_y <= upper_bound {
                let next_y = points[current + 1].y.floor() as usize;
                if next_y > upper_bound {
                    res = Some(current);
                    break;
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
    lower..=upper
}
