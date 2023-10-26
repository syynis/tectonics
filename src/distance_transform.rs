use vek::Vec2;

use crate::grid::Grid;

pub fn distance_transform(grid: &Grid<u8>) -> Grid<f64> {
    let dim: Vec2<usize> = grid.size().as_();
    let horizontal = horizontal_distance_transform(grid);

    let vertical_scan = |x, y| -> f64 {
        let total_edt = (0..dim.y).map(|y2| {
            let horz_val: f64 = horizontal[x + y2 * dim.x];
            (y2 as f64 - y as f64).powi(2) + horz_val.powi(2)
        });
        total_edt
            .reduce(f64::min)
            .unwrap()
            .min((y as f64).powi(2))
            .min(((dim.y - y) as f64).powi(2))
    };

    Grid::populate_from(grid.size(), |pos| {
        vertical_scan(pos.x as usize, pos.y as usize)
    })
}

fn horizontal_distance_transform(grid: &Grid<u8>) -> Vec<f64> {
    let dim: Vec2<usize> = grid.size().as_();
    let len = grid.size().product() as usize;
    let mut horizontal = grid
        .iter()
        .map(|(_, elem)| if *elem != 0 { len as f64 } else { 0. })
        .collect::<Vec<f64>>();

    let scan = |x, y, min: &mut f64, horizontal: &mut Vec<f64>| {
        let idx = x + y * dim.x;
        let f: f64 = horizontal[idx];
        let next = *min + 1.;
        let v = f.min(next);
        horizontal[idx] = v;
        *min = v;
    };

    for y in 0..dim.y {
        let mut min = 0.;
        for x in 0..dim.x {
            scan(x, y, &mut min, &mut horizontal);
        }
        min = 0.;
        for x in (0..dim.x).rev() {
            scan(x, y, &mut min, &mut horizontal);
        }
    }

    horizontal
}
