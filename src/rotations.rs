use std::collections::VecDeque;

use crate::moves::SeveralMoves;

fn get_move_place(mv: &str) -> Option<usize> {
    let mv = match mv.strip_prefix('-') {
        Some(mv) => mv,
        None => mv,
    };
    Some(
        2 * match mv {
            "f0" => 0,
            "f2" => 1,
            "d0" => 2,
            "d2" => 3,
            "r0" => 4,
            "r2" => 5,
            _ => return None,
        },
    )
}

pub fn apply_rotation(rot: usize, mv: &str, rev: bool) -> usize {
    let mut shift = if mv.starts_with('-') { 3 } else { 1 };
    if rev {
        shift = 4 - shift;
    }
    if let Some(place) = get_move_place(mv) {
        let now = (rot >> place) & 3;
        let new = (now + shift) & 3;
        rot ^ (now << place) ^ (new << place)
    } else {
        rot
    }
}

pub fn apply_rotations(rot: usize, moves: &SeveralMoves) -> usize {
    let mut res = rot;
    for small_mv in moves.name.iter() {
        res = apply_rotation(res, small_mv, false);
    }
    res
}

pub fn apply_rotations_rev(rot: usize, moves: &SeveralMoves) -> usize {
    let mut res = rot;
    for small_mv in moves.name.iter() {
        res = apply_rotation(res, small_mv, true);
    }
    res
}

pub fn get_rotations_dists(prev_moves: &[SeveralMoves], next_moves: &[SeveralMoves]) -> Vec<usize> {
    let mut res = vec![usize::MAX; 1 << 12];
    res[0] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(0);
    while let Some(rot) = queue.pop_front() {
        for mv in next_moves.iter() {
            let new_rot = apply_rotations(rot, mv);
            let ndist = res[rot];
            if res[new_rot] > ndist {
                res[new_rot] = ndist;
                queue.push_back(new_rot);
            }
        }
    }
    for v in 0..res.len() {
        if res[v] == 0 {
            queue.push_back(v);
        }
    }
    while let Some(rot) = queue.pop_front() {
        for mv in prev_moves.iter() {
            let new_rot = apply_rotations_rev(rot, mv);
            let ndist = res[rot] + mv.name.len();
            if res[new_rot] > ndist {
                res[new_rot] = ndist;
                queue.push_back(new_rot);
            }
        }
    }
    res
}

pub fn conv_rotations(rot: usize) -> Vec<usize> {
    let mut res = vec![];
    for i in 0..6 {
        res.push((rot >> (2 * i)) & 3);
    }
    res
}
