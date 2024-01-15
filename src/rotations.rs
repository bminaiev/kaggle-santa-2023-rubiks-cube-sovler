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

pub fn apply_rotation(rot: usize, mv: &str) -> usize {
    let shift = if mv.starts_with('-') { 3 } else { 1 };
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
        res = apply_rotation(res, small_mv);
    }
    res
}

pub fn get_rotations_dists(prev_moves: &[SeveralMoves], next_moves: &[SeveralMoves]) -> Vec<usize> {
    let mut res = vec![usize::MAX; 1 << 12];
    res[0] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(0);
    while let Some(rot) = queue.pop_front() {
        for i in 0..2 {
            let moves = if i == 0 { next_moves } else { prev_moves };
            for mv in moves.iter() {
                let new_rot = apply_rotations(rot, mv);
                let ndist = res[rot] + (if i == 0 { 0 } else { mv.name.len() });
                if res[new_rot] > ndist {
                    res[new_rot] = ndist;
                    queue.push_back(new_rot);
                }
            }
        }
    }
    res
}
