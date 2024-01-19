use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    mem,
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    data::Data, permutation::Permutation, sol_utils::TaskSolution, solutions_log::SolutionsLog,
};

///
///  AAAAA BBBBB
/// A     C     B
/// A    B A    B
/// A    B A    B
/// A    B A    B
/// A     C     B
///  AAAAA BBBBB
fn show_state(sol: &TaskSolution, colors: &[u8]) {
    let moves = &sol.task.info.moves;
    let left_cycle = &moves[&"l".to_string()].cycles[0];
    let right_cycle = &moves[&"r".to_string()].cycles[0];
    let n = left_cycle.len();
    let same_id = (1..n)
        .find(|x| left_cycle.contains(x) && right_cycle.contains(x))
        .unwrap();
    let mid_h = same_id;
    let rows = mid_h + 4;
    let mid_w = (n - mid_h * 2 - 4) / 2 + 1;
    let cols = mid_w * 2 + 3;
    let mut field = vec![vec![u8::MAX; cols]; rows];
    for cycle_id in 0..2 {
        for i in 0..n {
            let h;
            let mut w;
            if i == 0 {
                h = mid_h + 2;
                w = mid_w + 1;
            } else if i == same_id + cycle_id {
                h = 1;
                w = mid_w + 1;
            } else if i < same_id + cycle_id {
                h = mid_h + 2 - i;
                w = mid_w + 2;
            } else if i < same_id + mid_w + 1 + cycle_id {
                let offset = i - same_id - cycle_id;
                h = 0;
                w = mid_w + 1 - offset;
            } else if i < same_id + mid_w + mid_h + 3 + cycle_id {
                let offset = i - same_id - mid_w - 1 - cycle_id;
                h = offset + 1;
                w = 0;
            } else {
                let offset = i - same_id - mid_w - mid_h - 3 - cycle_id;
                h = mid_h + 3;
                w = offset + 1;
            }
            if cycle_id == 1 {
                w = cols - 1 - w;
            }
            let idx = if cycle_id == 0 {
                left_cycle[i]
            } else {
                right_cycle[(n - i) % n]
            };
            let color = colors[idx];

            field[h][w] = color;
        }
    }
    for h in 0..field.len() {
        for w in 0..field[h].len() {
            if field[h][w] == u8::MAX {
                print!(" ");
            } else {
                let c = (b'A' + field[h][w] as u8) as char;
                print!("{}", c);
            }
        }
        println!();
    }
}

fn hash_state(a: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    a.hash(&mut hasher);
    hasher.finish()
}

struct Prev {
    prev_hash: u64,
    edge_id: Option<usize>,
}

fn solve_one_task(task: &mut TaskSolution) {
    let move_keys: Vec<String> = task.task.info.moves.keys().cloned().collect();
    let moves = task.task.info.moves.clone();
    let start_state: Vec<u8> = task
        .state
        .iter()
        .map(|&p| task.task.solution_state[p] as u8)
        .collect();
    let expected_state: Vec<u8> = task.task.solution_state.iter().map(|&x| x as u8).collect();

    let mut important_ids = vec![98, 97, 96, 95, 94, 93, 101, 102, 103, 104, 105, 106];
    let mut maybe_important_ids = vec![174, 175, 176, 21, 22, 23];
    if expected_state.len() < 100 {
        important_ids.clear();
        maybe_important_ids.clear();
    }
    // let mut tmp_state = vec![1; start_state.len()];
    // for i in 0..important_ids.len() {
    //     tmp_state[important_ids[i]] = 0;
    // }
    // show_state(task, &tmp_state);

    let score = |state: &[u8]| -> usize {
        let mut res = 0;
        for i in 0..state.len() {
            if state[i] != expected_state[i] {
                res += 1;
            }
        }
        for &id in important_ids.iter() {
            if state[id] != expected_state[id] {
                res += 10;
            }
        }
        for &id in maybe_important_ids.iter() {
            if state[id] != expected_state[id] {
                res += 1;
            }
        }
        res
    };

    let mut seen = HashMap::new();
    seen.insert(
        hash_state(&start_state),
        Prev {
            prev_hash: 0,
            edge_id: None,
        },
    );
    let mut queue = vec![Vec::new(); 1000];
    queue[0].push(start_state);
    const MAX_CHECK: usize = 787_788;
    let mut to_remove = VecDeque::new();
    for it in 0..queue.len() {
        let mut cur_lvl = vec![];
        mem::swap(&mut queue[it], &mut cur_lvl);
        cur_lvl.sort_by_cached_key(|x| score(x));
        for i in MAX_CHECK..cur_lvl.len() {
            let state = &cur_lvl[i];
            let hash = hash_state(state);
            to_remove.push_back(hash);
        }
        while to_remove.len() > 10_000_000 {
            let hash = to_remove.pop_front().unwrap();
            seen.remove(&hash);
        }
        cur_lvl.truncate(MAX_CHECK);
        {
            let state = &cur_lvl[0];
            let cur_score = score(state);
            eprintln!("Len = {it}. Score = {cur_score}");
            show_state(task, state);
            if cur_score <= task.task.num_wildcards {
                eprintln!("Found solution!");
                let mut cur_hash = hash_state(state);
                let mut moves = vec![];
                loop {
                    let prev = seen.get(&cur_hash).unwrap();
                    if let Some(edge_id) = prev.edge_id {
                        moves.push(move_keys[edge_id].clone());
                        cur_hash = prev.prev_hash;
                    } else {
                        break;
                    }
                }
                moves.reverse();
                for mv in moves.iter() {
                    task.append_move(mv);
                }
                assert!(task.is_solved_with_wildcards());
                eprintln!("Solved!");
                return;
            }
        }
        let chunk_size = 1 + cur_lvl.len() / 10;
        let chunks: Vec<_> = cur_lvl.chunks(chunk_size).collect();
        let next_edges: Vec<_> = chunks
            .par_iter()
            .flat_map(|states| {
                let mut res = vec![];
                let mut seen2 = HashSet::new();
                for state in states.iter() {
                    let prev_hash = hash_state(state);
                    for (edge_id, mv) in move_keys.iter().enumerate() {
                        let mv = moves.get(&mv.to_string()).unwrap();
                        let mut nstate = state.clone();
                        mv.apply(&mut nstate);
                        let hash = hash_state(&nstate);
                        if !seen.contains_key(&hash) && !seen2.contains(&hash) {
                            seen2.insert(hash);
                            res.push((
                                hash,
                                Prev {
                                    prev_hash,
                                    edge_id: Some(edge_id),
                                },
                                nstate,
                            ));
                        }
                    }
                }
                res
            })
            .collect();
        for (hash, prev, nstate) in next_edges {
            if !seen.contains_key(&hash) {
                seen.insert(hash, prev);
                queue[it + 1].push(nstate);
            }
        }
    }
}

fn find_patterns(task: &TaskSolution) {
    let moves = task.task.info.moves.clone();
    let mut hm = HashSet::new();

    let mut check = |mvs: &[&str]| {
        let mut perm = Permutation::identity();
        for mv in mvs {
            let mv = moves.get(&mv.to_string()).unwrap();
            perm = perm.combine(&mv);
        }
        let sum_len = perm.sum_len();
        if sum_len < 10 {
            if hm.insert(perm.clone()) {
                eprintln!("{mvs:?} -> {:?}", perm.cycles);
            }
        }
    };

    // check(&[]);
    // for k1 in moves.keys() {
    //     check(&[k1]);
    //     for k2 in moves.keys() {
    //         check(&[k1, k2]);
    //         for k3 in moves.keys() {
    //             check(&[k1, k2, k3]);
    //             for k4 in moves.keys() {
    //                 check(&[k1, k2, k3, k4]);
    //                 for k5 in moves.keys() {
    //                     for k6 in moves.keys() {
    //                         // check(&[k1, k2, k3, k4, k5, k6]);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

pub fn solve_wreath(data: &Data, log: &mut SolutionsLog) {
    let mut tasks = TaskSolution::all_by_type(data, "wreath_21/21", false);

    for task in tasks.iter_mut() {
        eprintln!("Task {}. Type: {}", task.task_id, task.task.puzzle_type);
        for (k, v) in task.task.info.moves.iter() {
            eprintln!("{} -> {:?}", k, v);
        }

        solve_one_task(task);

        log.append(task);
    }
}
