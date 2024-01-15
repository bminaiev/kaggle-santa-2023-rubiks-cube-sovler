use std::{
    collections::{hash_map::DefaultHasher, HashMap, VecDeque},
    hash::Hasher,
};

use crate::{
    moves::SeveralMoves,
    rotations::{apply_rotations, apply_rotations_rev, conv_rotations},
    utils::get_all_perms,
};

pub struct Groups {
    pub groups: Vec<Vec<Vec<usize>>>,
    pub by_elem: HashMap<Vec<usize>, usize>,
    pub blocks: Vec<Vec<usize>>,
}

pub fn get_groups(blocks: &[Vec<usize>], moves: &[SeveralMoves]) -> Groups {
    eprintln!("Get groups");
    let mut res = HashMap::new();
    let mut cnt_groups = 0;
    let mut all_groups = vec![];
    for block in blocks.iter() {
        for perm in get_all_perms(block) {
            if res.contains_key(&perm) {
                continue;
            }
            let mut cur_group = vec![];
            let mut queue = VecDeque::new();
            cur_group.push(perm.clone());
            queue.push_back(perm.clone());
            res.insert(perm, cnt_groups);
            while let Some(perm) = queue.pop_front() {
                for mov in moves.iter() {
                    let mut new_perm = perm.clone();
                    for x in new_perm.iter_mut() {
                        *x = mov.permutation.next(*x);
                    }
                    if !res.contains_key(&new_perm) {
                        cur_group.push(new_perm.clone());
                        res.insert(new_perm.clone(), cnt_groups);
                        queue.push_back(new_perm);
                    }
                }
            }
            all_groups.push(cur_group);
            cnt_groups += 1;
        }
    }
    Groups {
        groups: all_groups,
        by_elem: res,
        blocks: blocks.to_vec(),
    }
}

pub fn blocks_bfs(start_block: &[usize], moves: &[SeveralMoves]) -> HashMap<Vec<usize>, usize> {
    let mut res = HashMap::new();
    let mut queue = VecDeque::new();
    queue.push_back(start_block.to_vec());
    res.insert(start_block.to_vec(), 0);
    while let Some(block) = queue.pop_front() {
        let cur_d = *res.get(&block).unwrap();
        for mov in moves.iter() {
            let mut new_block = block.clone();
            for x in new_block.iter_mut() {
                *x = mov.permutation.next(*x);
            }
            let ndist = cur_d + mov.name.len();
            if !res.contains_key(&new_block) || res[&new_block] > ndist {
                res.insert(new_block.clone(), ndist);
                queue.push_back(new_block);
            }
        }
    }
    res
}

impl Groups {
    pub fn hash(&self, a: &[usize]) -> DefaultHasher {
        let mut hasher = DefaultHasher::new();
        for block in self.blocks.iter() {
            let mut group = vec![];
            for &x in block.iter() {
                group.push(a[x]);
            }
            let group_id = *self.by_elem.get(&group).unwrap();
            hasher.write_usize(group_id);
        }
        hasher
    }
}

#[derive(Clone)]
pub struct Edge {
    pub next_state_hash: u64,
    pub mov_idx: usize,
    pub len: usize,
}

pub const PREC_LIMIT: usize = 100_000_000;

fn join_hash(a: u64, b: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(a);
    hasher.write_usize(b);
    hasher.finish()
}

pub fn precompute_moves_from_final_state(
    moves: &[SeveralMoves],
    get_state: &impl Fn(&[usize], bool) -> u64,
    limit: usize,
    final_state: Vec<usize>,
    include_rotations: bool,
) -> HashMap<u64, Edge> {
    let get_state = |a: &[usize], rot: usize| {
        let mut hash = get_state(a, false);
        if include_rotations {
            hash = join_hash(hash, rot);
        }
        hash
    };

    let hash = get_state(&final_state, 0);
    let mut queues = vec![Vec::new(); 50];
    queues[0].push((final_state, 0));
    let mut res = HashMap::new();
    res.insert(
        hash,
        Edge {
            next_state_hash: hash,
            mov_idx: usize::MAX,
            len: 0,
        },
    );
    let mut it = 0;
    for cur_d in 0..queues.len() {
        eprintln!("cur_d: {cur_d}. it = {it}");
        while let Some((state, rot)) = queues[cur_d].pop() {
            it += 1;
            if it % 100_000 == 0 {
                eprintln!("it: {}", it);
            }
            if it > limit {
                eprintln!("LIMIT {limit} REACHED");
                return res;
            }
            let cur_hash = get_state(&state, rot);
            for (mov_idx, mov) in moves.iter().enumerate() {
                let mut new_state = state.clone();
                mov.permutation.apply_rev(&mut new_state);
                let new_rot = apply_rotations_rev(rot, mov);
                let hash = get_state(&new_state, new_rot);
                let len = cur_d + mov.name.len();
                let should_add = match res.get(&hash) {
                    None => true,
                    Some(edge) => edge.len > len,
                };
                if should_add {
                    res.insert(
                        hash,
                        Edge {
                            next_state_hash: cur_hash,
                            mov_idx,
                            len,
                        },
                    );
                    queues[len].push((new_state, new_rot));
                }
            }
        }
        queues[cur_d].shrink_to_fit();
    }
    res
}

pub fn precompute_moves(
    n: usize,
    moves: &[SeveralMoves],
    get_state: &mut impl Fn(&[usize], bool) -> u64,
    limit: usize,
    include_rotations: bool,
) -> HashMap<u64, Edge> {
    precompute_moves_from_final_state(moves, get_state, limit, (0..n).collect(), include_rotations)
}

pub fn apply_precomputed_moves(
    state: &mut [usize],
    prec: &HashMap<u64, Edge>,
    get_state: impl Fn(&[usize], bool) -> u64,
    answer: &mut Vec<String>,
    possible_moves: &[SeveralMoves],
    mut cur_rot: Option<usize>,
) -> bool {
    let mut cur_hash = get_state(state, false);
    if let Some(cur_rot) = cur_rot {
        cur_hash = join_hash(cur_hash, cur_rot);
    }
    eprintln!("Start rot: {cur_rot:?}. Start hash: {cur_hash}");
    while let Some(edge) = prec.get(&cur_hash) {
        if edge.mov_idx == usize::MAX {
            return true;
        }
        let mov = &possible_moves[edge.mov_idx];
        mov.permutation.apply(state);
        if let Some(cur_rot) = &mut cur_rot {
            *cur_rot = apply_rotations(*cur_rot, mov);
        }
        answer.extend(mov.name.clone());
        cur_hash = edge.next_state_hash;
        let mut real_state_hash = get_state(state, false);
        if let Some(cur_rot) = cur_rot {
            real_state_hash = join_hash(real_state_hash, cur_rot);
            eprintln!(
                "Apply move... New rot: {cur_rot}. Moves: {:?}. Rot: {:?}",
                mov.name,
                conv_rotations(cur_rot)
            );
        }
        assert_eq!(cur_hash, real_state_hash);
    }
    false
}

pub fn apply_precomputed_moves_bfs(
    start_state: &mut [usize],
    prec: &HashMap<u64, Edge>,
    get_state: impl Fn(&[usize], bool) -> u64,
    answer: &mut Vec<String>,
    moves: &[SeveralMoves],
) -> bool {
    let start_hash = get_state(start_state, false);
    let mut queues = vec![Vec::new(); 50];
    queues[0].push(start_state.to_vec());
    let mut res = HashMap::new();
    res.insert(
        start_hash,
        Edge {
            next_state_hash: start_hash,
            mov_idx: usize::MAX,
            len: 0,
        },
    );
    let mut it = 0;
    for cur_d in 0..queues.len() {
        while let Some(state) = queues[cur_d].pop() {
            it += 1;
            if it % 100_000 == 0 {
                eprintln!("it: {}", it);
            }
            let cur_hash = get_state(&state, false);
            for (mov_idx, mov) in moves.iter().enumerate() {
                let mut new_state = state.clone();
                mov.permutation.apply(&mut new_state);
                let hash = get_state(&new_state, false);
                let len = cur_d + mov.name.len();

                if prec.contains_key(&hash) {
                    eprintln!("FOUND HIT AT DISTANCE {cur_d}");
                    let mut moves_to_apply = vec![mov];
                    let mut tmp_hash = cur_hash;
                    while tmp_hash != start_hash {
                        let edge = res.get(&tmp_hash).unwrap();
                        moves_to_apply.push(&moves[edge.mov_idx]);
                        tmp_hash = edge.next_state_hash;
                    }
                    moves_to_apply.reverse();
                    for mov in moves_to_apply.iter() {
                        mov.permutation.apply(start_state);
                        answer.extend(mov.name.clone());
                    }
                    assert_eq!(get_state(start_state, false), hash);
                    assert!(apply_precomputed_moves(
                        start_state,
                        prec,
                        &get_state,
                        answer,
                        moves,
                        None
                    ));
                    let st1 = get_state(&(0..start_state.len()).collect::<Vec<_>>(), true);
                    let st2 = get_state(start_state, true);
                    eprintln!("ST1: {}", st1);
                    eprintln!("ST2: {}", st2);
                    assert_eq!(st1, st2);
                    return true;
                }

                let should_add = match res.get(&hash) {
                    None => true,
                    Some(edge) => edge.len > len,
                };
                if should_add {
                    res.insert(
                        hash,
                        Edge {
                            next_state_hash: cur_hash,
                            mov_idx,
                            len,
                        },
                    );
                    queues[len].push(new_state);
                }
            }
        }
    }
    false
}
