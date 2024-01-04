use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet, VecDeque},
    hash::Hasher,
};

use crate::{
    checker::check_solution,
    data::Data,
    moves::{create_moves, SeveralMoves},
    puzzle_type::PuzzleType,
    utils::{get_all_perms, get_blocks, get_start_permutation},
};

// Thistlethwaite's groups
fn create_move_groups(puzzle_info: &PuzzleType) -> Vec<Vec<SeveralMoves>> {
    let mut move_groups = vec![];
    move_groups.push(create_moves(
        puzzle_info,
        &["d0", "d1", "d2", "f0", "f1", "f2", "r0", "r1", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0", "d2", "f0", "f2", "r0", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0x2", "d2x2", "f0", "f2", "r0", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0x2", "d2x2", "f0x2", "f2x2", "r0", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0x2", "d2x2", "f0x2", "f2x2", "r0x2", "r2x2"],
    ));
    move_groups.push(create_moves(puzzle_info, &[]));
    move_groups
}

#[derive(Clone)]
struct Edge<'a> {
    next_state_hash: u64,
    mov: Option<&'a SeveralMoves>,
    len: usize,
}

fn precompute_moves(
    n: usize,
    moves: &[SeveralMoves],
    get_state: impl Fn(&[usize]) -> u64,
) -> HashMap<u64, Edge<'_>> {
    let final_state: Vec<_> = (0..n).collect();
    let hash = get_state(&final_state);
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
    for cur_d in 0..queues.len() {
        while let Some(state) = queues[cur_d].pop() {
            let cur_hash = get_state(&state);
            for mov in moves.iter() {
                let mut new_state = state.clone();
                mov.permutation.apply_rev(&mut new_state);
                let hash = get_state(&new_state);
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

pub fn solve(data: &Data, task_id: usize) {
    eprintln!("SOLVING {task_id}");
    let task = &data.puzzles[task_id];
    let example_solution = &data.solutions[&task_id];
    let mut state = get_start_permutation(task, example_solution);
    eprintln!("START STATE: {:?}", state);

    let move_groups = create_move_groups(&task.info);
    eprintln!("Total move groups: {}", move_groups.len());
    let moves = task.info.moves.values().cloned().collect::<Vec<_>>();
    let blocks = get_blocks(task, &moves);
    // let n = task.info.n;
    let mut answer = vec![];
    for w in move_groups.windows(2) {
        eprintln!("Calculating groups...");
        let groups = get_groups(&blocks, &w[1]);
        eprintln!("Groups are calculated.");
        let calc_hash = |a: &[usize]| {
            let mut hasher = DefaultHasher::new();
            // let mut inv = vec![0; n];
            // for (i, &x) in a.iter().enumerate() {
            //     inv[x] = i;
            // }
            // let mut ids = vec![];
            for block in blocks.iter() {
                let mut group = vec![];
                for &x in block.iter() {
                    group.push(a[x]);
                }
                let group_id = *groups.get(&group).unwrap();
                // ids.push(group_id);
                hasher.write_usize(group_id);
            }
            hasher.finish()
        };
        let prec = precompute_moves(task.info.n, &w[0], calc_hash);
        eprintln!("Precumputed size: {}", prec.len());
        apply_precomputed_moves(&mut state, &prec, calc_hash, &mut answer);
        eprintln!("New answer: {:?}. Len = {}", answer, answer.len());
        eprintln!("new state: {:?}", state);
    }
    for i in 0..state.len() {
        assert_eq!(i, state[i]);
    }
    check_solution(task, &answer);
    eprintln!("LAST STATE: {}", task.convert_solution(&answer));
}

fn apply_precomputed_moves(
    state: &mut [usize],
    prec: &HashMap<u64, Edge<'_>>,
    get_state: impl Fn(&[usize]) -> u64,
    answer: &mut Vec<String>,
) {
    let mut cur_hash = get_state(state);
    while let Some(edge) = prec.get(&cur_hash) {
        if let Some(mov) = edge.mov {
            mov.permutation.apply(state);
            answer.extend(mov.name.clone());
        } else {
            return;
        }
        cur_hash = edge.next_state_hash;
    }
    unreachable!();
}

fn get_groups(blocks: &[Vec<usize>], moves: &[SeveralMoves]) -> HashMap<Vec<usize>, usize> {
    let mut res = HashMap::new();
    let mut cnt_groups = 0;
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
            cnt_groups += 1;
        }
    }
    res
}
