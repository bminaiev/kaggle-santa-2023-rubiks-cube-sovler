use std::collections::HashSet;

use crate::utils::show_cube_ids;

#[derive(Clone, Copy)]
enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

fn get_square_side(sq: &[Vec<usize>], side: Side) -> Vec<usize> {
    let n = sq.len();
    match side {
        Side::Top => sq[0].clone(),
        Side::Bottom => sq[n - 1].clone(),
        Side::Left => sq.iter().map(|row| row[0]).collect(),
        Side::Right => sq.iter().map(|row| row[n - 1]).collect(),
    }
}

#[derive(Clone, Copy)]
struct SameEdge {
    sq1: usize,
    sq2: usize,
    side1: Side,
    side2: Side,
    rev: bool,
}

fn get_same_edges() -> Vec<SameEdge> {
    let e = |sq1: usize, sq2: usize, side1: Side, side2: Side, rev: bool| SameEdge {
        sq1,
        sq2,
        side1,
        side2,
        rev,
    };
    vec![
        e(0, 1, Side::Bottom, Side::Top, false),
        e(0, 2, Side::Right, Side::Top, true),
        e(0, 4, Side::Left, Side::Top, false),
        e(0, 3, Side::Top, Side::Top, true),
        e(1, 2, Side::Right, Side::Left, false),
        e(1, 4, Side::Left, Side::Right, false),
        e(1, 5, Side::Bottom, Side::Top, false),
        e(2, 3, Side::Right, Side::Left, false),
        e(2, 5, Side::Bottom, Side::Right, false),
        e(3, 4, Side::Right, Side::Left, false),
        e(3, 5, Side::Bottom, Side::Bottom, true),
        e(4, 5, Side::Bottom, Side::Left, true),
    ]
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub pos1: usize,
    pub pos2: usize,
}

pub fn calc_cube_edges(squares: &[Vec<Vec<usize>>]) -> Vec<Vec<[Edge; 2]>> {
    let n = squares[0].len();
    // 3 -> 0
    // 4 -> 1
    // 5 -> 1
    // 6 -> 2
    let mut res = vec![vec![]; (n - 2) / 2];

    // 3 -> (1, 1)
    // 4 -> (1, 2)
    let center1 = (n - 1) / 2;
    let center2 = n / 2;

    let same_edges = get_same_edges();
    for same_edge in same_edges.iter() {
        let sq1 = &squares[same_edge.sq1];
        let sq2 = &squares[same_edge.sq2];
        let side1 = get_square_side(sq1, same_edge.side1);
        let mut side2 = get_square_side(sq2, same_edge.side2);
        if same_edge.rev {
            side2.reverse();
        }
        for d in 0..res.len() {
            let positions = if n % 2 == 0 {
                if d == 0 {
                    vec![(center1, center2)]
                } else {
                    vec![
                        (center1 - d + 1, center1 - d),
                        (center2 + d - 1, center2 + d),
                    ]
                }
            } else {
                vec![
                    (center1 - d, center1 - d - 1),
                    (center2 + d, center2 + d + 1),
                ]
            };
            for &(pos1, pos2) in positions.iter() {
                let edge1 = Edge {
                    pos1: side1[pos1],
                    pos2: side2[pos1],
                };
                let edge2 = Edge {
                    pos1: side1[pos2],
                    pos2: side2[pos2],
                };
                res[d].push([edge1, edge2]);
            }
        }
    }
    res
}

pub fn build_squares(sz: usize) -> Vec<Vec<Vec<usize>>> {
    let mut res = vec![];
    for i in 0..6 {
        let offset = i * sz * sz;
        let mut sq = vec![vec![0; sz]; sz];
        for r in 0..sz {
            for c in 0..sz {
                sq[r][c] = offset + r * sz + c;
            }
        }
        res.push(sq);
    }
    res
}

pub fn calc_edges_score(
    edges: &[Vec<[Edge; 2]>],
    state: &[usize],
    target_state: &[usize],
) -> Vec<usize> {
    let norm = |state: &[usize], edges: &[Edge; 2]| -> [usize; 4] {
        let c1 = state[edges[0].pos1];
        let c2 = state[edges[0].pos2];
        let c3 = state[edges[1].pos1];
        let c4 = state[edges[1].pos2];
        if c1 < c2 {
            [c1, c2, c3, c4]
        } else {
            [c2, c1, c4, c3]
        }
    };
    let mut res = vec![];
    for lvl in 0..edges.len() {
        let mut expect = HashSet::new();
        for edge in edges[lvl].iter() {
            expect.insert(norm(target_state, edge));
        }
        let mut cnt = 0;
        for edge in edges[lvl].iter() {
            if expect.contains(&norm(state, edge)) {
                cnt += 1;
            }
        }
        res.push(cnt);
    }
    res
}

pub fn viz_edges_score(
    edges: &[Vec<[Edge; 2]>],
    state: &[usize],
    target_state: &[usize],
) -> Vec<usize> {
    let norm = |state: &[usize], edges: &[Edge; 2]| -> [usize; 4] {
        let c1 = state[edges[0].pos1];
        let c2 = state[edges[0].pos2];
        let c3 = state[edges[1].pos1];
        let c4 = state[edges[1].pos2];
        if c1 < c2 {
            [c1, c2, c3, c4]
        } else {
            [c2, c1, c4, c3]
        }
    };
    let mut res = vec![];
    let mut good_ids = vec![];
    for lvl in 0..edges.len() {
        let mut expect = HashSet::new();
        for edge in edges[lvl].iter() {
            expect.insert(norm(target_state, edge));
        }
        let mut cnt = 0;
        for edge in edges[lvl].iter() {
            if expect.contains(&norm(state, edge)) {
                good_ids.push(edge[0].pos1);
                good_ids.push(edge[0].pos2);
                good_ids.push(edge[1].pos1);
                good_ids.push(edge[1].pos2);
                eprintln!("Good edge: {edge:?}");
            }
        }
        res.push(cnt);
    }
    good_ids
}

pub fn calc_cube_centers(squares: &[Vec<Vec<usize>>]) -> Vec<bool> {
    let n = squares[0].len();
    let mut res = vec![true; 6 * n * n];
    for sq in squares.iter() {
        for r in 0..n {
            for c in 0..n {
                if r == 0 || r == n - 1 || c == 0 || c == n - 1 {
                    res[sq[r][c]] = false;
                }
            }
        }
    }
    res
}

#[test]
fn test() {
    let sz = 5;
    let squares = build_squares(sz);
    let edges = calc_cube_edges(&squares);
    for lvl in 0..edges.len() {
        eprintln!("lvl={}", lvl);
        for edge in edges[lvl].iter() {
            eprintln!("{:?} {:?}", edge[0], edge[1]);
            let ids = vec![edge[0].pos1, edge[0].pos2, edge[1].pos1, edge[1].pos2];
            show_cube_ids(&ids, sz);
        }
    }
}
