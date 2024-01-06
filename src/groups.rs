use std::{
    collections::{hash_map::DefaultHasher, HashMap, VecDeque},
    hash::Hasher,
};

use crate::{moves::SeveralMoves, utils::get_all_perms};

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
pub struct Edge<'a> {
    pub next_state_hash: u64,
    pub mov: Option<&'a SeveralMoves>,
    pub len: usize,
}

pub fn precompute_moves<'a>(
    n: usize,
    moves: &'a [SeveralMoves],
    get_state: &mut impl FnMut(&[usize], bool) -> u64,
) -> HashMap<u64, Edge<'a>> {
    let final_state: Vec<_> = (0..n).collect();
    let hash = get_state(&final_state, false);
    let mut queues = vec![Vec::new(); 50];
    queues[0].push(final_state);
    let mut res = HashMap::new();
    res.insert(
        hash,
        Edge {
            next_state_hash: hash,
            mov: None,
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
            for mov in moves.iter() {
                let mut new_state = state.clone();
                mov.permutation.apply_rev(&mut new_state);
                let hash = get_state(&new_state, false);
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
                            mov: Some(mov),
                            len,
                        },
                    );
                    queues[len].push(new_state);
                }
            }
        }
    }
    res
}

pub fn apply_precomputed_moves(
    state: &mut [usize],
    prec: &HashMap<u64, Edge<'_>>,
    get_state: impl Fn(&[usize], bool) -> u64,
    answer: &mut Vec<String>,
) -> bool {
    let mut cur_hash = get_state(state, false);
    while let Some(edge) = prec.get(&cur_hash) {
        if let Some(mov) = edge.mov {
            mov.permutation.apply(state);
            answer.extend(mov.name.clone());
        } else {
            return true;
        }
        cur_hash = edge.next_state_hash;
    }
    false
}
