use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    hash::Hasher,
};

use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

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

fn precompute_moves<'a>(
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

#[derive(Clone)]
struct TaskSolution {
    task_id: usize,
    answer: Vec<String>,
    failed_on_stage: Option<usize>,
    state: Vec<usize>,
}

type Block = Vec<usize>;

pub fn solve(data: &Data, task_type: &str) {
    eprintln!("SOLVING {task_type}");
    let mut solutions = vec![];
    for task in data.puzzles.iter() {
        if task.puzzle_type == task_type {
            solutions.push(TaskSolution {
                task_id: task.id,
                answer: vec![],
                failed_on_stage: None,
                state: get_start_permutation(task, &data.solutions[&task.id]),
            })
        }
    }
    assert!(!solutions.is_empty());
    let puzzle_info = data.puzzle_info.get(task_type).unwrap();

    let move_groups = create_move_groups(puzzle_info);
    eprintln!("Total move groups: {}", move_groups.len());
    let moves = puzzle_info.moves.values().cloned().collect::<Vec<_>>();
    let blocks = get_blocks(puzzle_info.n, &moves);

    for (i, block) in blocks.iter().enumerate() {
        eprintln!("Block {i}: {block:?}");
    }

    for (step, w) in move_groups.windows(2).enumerate() {
        eprintln!("Calculating groups... Step: {step}");
        let groups = get_groups(&blocks, &w[1]);
        eprintln!("Groups are calculated.");

        let hacks_info = step == 3;

        let mut calc_hash = |a: &[usize], debug: bool| {
            let mut hasher = DefaultHasher::new();
            for block in blocks.iter() {
                let mut group = vec![];
                for &x in block.iter() {
                    group.push(a[x]);
                }
                let group_id = *groups.by_elem.get(&group).unwrap();
                hasher.write_usize(group_id);
            }
            if hacks_info {
                for block in blocks.iter() {
                    if block.len() != 3 {
                        continue;
                    }
                    let min = block.iter().map(|&x| a[x]).min().unwrap();
                    hasher.write_usize(min);
                }
            }
            hasher.finish()
        };
        let prec = precompute_moves(puzzle_info.n, &w[0], &mut calc_hash);
        eprintln!("Precumputed size: {}", prec.len());
        let mut cnt_ok = 0;
        for sol in solutions.iter_mut() {
            if sol.failed_on_stage.is_some() {
                continue;
            }
            if !apply_precomputed_moves(&mut sol.state, &prec, calc_hash, &mut sol.answer) {
                sol.failed_on_stage = Some(step);
            } else {
                cnt_ok += 1;
            }
        }
        eprintln!("Still ok solutions: {cnt_ok}/{}", solutions.len());
    }
    for sol in solutions.iter() {
        if sol.failed_on_stage.is_some() {
            continue;
        }
        let task = &data.puzzles[sol.task_id];
        eprintln!(
            "TASK: {}. SOL LEN={}, colors = {}, State={}",
            sol.task_id,
            sol.answer.len(),
            task.num_colors,
            task.convert_solution(&sol.answer)
        );
        check_solution(task, &sol.answer);
    }
}

fn apply_precomputed_moves(
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

struct Groups {
    groups: Vec<Vec<Vec<usize>>>,
    by_elem: HashMap<Vec<usize>, usize>,
}

fn get_groups(blocks: &[Vec<usize>], moves: &[SeveralMoves]) -> Groups {
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
    }
}
