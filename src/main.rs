use std::{
    cmp::Reverse,
    collections::{hash_map::DefaultHasher, BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    os::linux::raw::stat,
    time::Instant,
};

use rand::{seq::SliceRandom, Rng};

use crate::{
    data::{load_data, Data},
    permutation::Permutation,
    puzzle::Puzzle,
    puzzle_type::PuzzleType,
    solver::solve,
    utils::get_blocks,
};

pub mod data;
pub mod permutation;
pub mod puzzle;
pub mod puzzle_type;
pub mod solver;
pub mod utils;

fn check_solution(task: &Puzzle, solution: &[String], puzzle_info: &BTreeMap<String, PuzzleType>) {
    let mut state = task.initial_state.clone();
    let puzzle_info = &puzzle_info[&task.puzzle_type];
    for step in solution.iter() {
        let perm = &puzzle_info.moves[step];
        for cycle in perm.cycles.iter() {
            for w in cycle.windows(2) {
                state.swap(w[0], w[1]);
            }
        }
    }
    let mut cnt_fails = 0;
    for i in 0..state.len() {
        if state[i] != task.solution_state[i] {
            cnt_fails += 1;
        }
    }
    println!("{}: {}/{} fails", task.id, cnt_fails, task.num_wildcards);
}

fn calc_hash(a: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    a.hash(&mut hasher);
    hasher.finish()
}

fn check_solution3(task: &Puzzle, solution: &[String], puzzle_info: &BTreeMap<String, PuzzleType>) {
    let mut state = task.initial_state.clone();
    let puzzle_info = &puzzle_info[&task.puzzle_type];
    let mut seen = HashMap::<u64, usize>::new();
    let mut saved = 0;
    let mut oks = vec![];
    let every = solution.len() / 100 + 1;
    for (step_it, step) in solution.iter().enumerate() {
        let perm = &puzzle_info.moves[step];
        for cycle in perm.cycles.iter() {
            for w in cycle.windows(2) {
                state.swap(w[0], w[1]);
            }
        }
        if step_it % every == 0 || step_it + 100 > solution.len() {
            let cnt_ok = state
                .iter()
                .zip(task.solution_state.iter())
                .filter(|(a, b)| a == b)
                .count();
            oks.push(cnt_ok);
        }
    }

    let sz = puzzle_info.moves.len();
    let mut moves_ids = HashMap::new();
    let mut move_names = vec![];
    for (pos, k) in puzzle_info.moves.keys().enumerate() {
        moves_ids.insert(k, pos);
        move_names.push(k);
    }
    if sz < 50 {
        // let mut next = vec![vec![0; sz]; sz];
        // for w in solution.windows(2) {
        //     let mv1 = moves_ids[&w[0]];
        //     let mv2 = moves_ids[&w[1]];
        //     next[mv1][mv2] += 1;
        // }
        // for i in 0..next.len() {
        //     print!("{:4} ", move_names[i]);
        //     for j in 0..next.len() {
        //         print!("{:3} ", next[i][j]);
        //     }
        //     println!();
        // }
        // println!();
    }

    let mut cnt_fails = 0;
    for i in 0..state.len() {
        if state[i] != task.solution_state[i] {
            cnt_fails += 1;
        }
    }
    println!(
        "{}: {}/{} fails. Saved: {saved}",
        task.id, cnt_fails, task.num_wildcards
    );
    println!("OKs: {:?}", oks);
}

fn check_solution2(task: &Puzzle, solution: &[String], puzzle_info: &BTreeMap<String, PuzzleType>) {
    let mut state = task.initial_state.clone();
    let puzzle_info = &puzzle_info[&task.puzzle_type];
    let mut tot_perm = Permutation::identity();
    for step in solution.iter() {
        let perm = &puzzle_info.moves[step];
        tot_perm = tot_perm.combine(perm);
    }
    tot_perm.apply(&mut state);

    let mut cnt_fails = 0;
    for i in 0..state.len() {
        if state[i] != task.solution_state[i] {
            cnt_fails += 1;
        }
    }
    println!("{}: {}/{} fails", task.id, cnt_fails, task.num_wildcards);
}

fn show_info(data: &Data) {
    let puzzle_info = &data.puzzle_info;
    for (k, v) in puzzle_info.iter() {
        println!("{}: n={}, cnt_moves={}", k, v.n, v.moves.len());
    }

    let puzzles = &data.puzzles;
    let solutions = &data.solutions;

    let mut sorted_test_ids = puzzles.iter().map(|p| p.id).collect::<Vec<_>>();
    sorted_test_ids.sort_by_key(|id| Reverse(solutions[id].len()));
    sorted_test_ids.reverse();

    let mut sum_lens = 0;
    for puzzle_id in sorted_test_ids[..].iter() {
        let puzzle = &puzzles[*puzzle_id];
        let sol = &solutions[&puzzle.id];
        println!(
            "{}: {}. Len: {}. Sol len: {}. Colors = {}",
            puzzle.id,
            puzzle.puzzle_type,
            puzzle.solution_state.len(),
            sol.len(),
            puzzle.num_colors
        );
        sum_lens += sol.len();
        check_solution3(puzzle, sol, &puzzle_info);
    }
    println!("Total len: {}", sum_lens);
}

fn rev(s: &str) -> String {
    if s.starts_with('-') {
        s[1..].to_string()
    } else {
        format!("-{}", s)
    }
}

fn analyze_puzzle_type(data: &Data, puzzle_type: &str) {
    let puzzle_info = &data.puzzle_info[puzzle_type];
    eprintln!("N = {}. Moves = {}", puzzle_info.n, puzzle_info.moves.len());

    // for mov in puzzle_info.moves.values() {
    //     eprint!("{}: ", mov.cycles.len());
    //     for cycle in mov.cycles.iter() {
    //         eprint!("{} ", cycle.len());
    //     }
    //     eprintln!();
    // }

    let mut queue = vec![];
    let mut dist = HashMap::new();
    for m in puzzle_info.moves.values() {
        queue.push(m.clone());
        dist.insert(m.clone(), 1);
    }

    let is_interesting = |m: &Permutation, seen: &HashMap<Permutation, usize>, ndist: usize| {
        let len = m.sum_len();
        len <= 10 && len != 0 && seen.get(m).unwrap_or(&usize::MAX) > &ndist
    };

    // let mut it1 = 0;
    // while it1 < queue.len() && queue.len() < 50000 {
    //     let m1 = queue[it1].clone();
    //     it1 += 1;
    //     for it2 in 0..it1 {
    //         let m2 = queue[it2].clone();
    //         for (m1, m2) in [(&m1, &m2), (&m2, &m1)].iter() {
    //             let new_m = m1.combine(m2).combine(&m1.inv()).combine(&m2.inv());
    //             let ndist = (dist[m1] + dist[m2]) * 2;
    //             if is_interesting(&new_m, &dist, ndist) {
    //                 if new_m.sum_len() < 6 {
    //                     eprintln!(
    //                         "{}/{}. Len = {}. {:?}",
    //                         it1,
    //                         queue.len(),
    //                         new_m.sum_len(),
    //                         new_m.cycles
    //                     );
    //                 }
    //                 queue.push(new_m.clone());
    //                 dist.insert(new_m, ndist);
    //             }
    //         }
    //     }
    // }
    println!("DONE!");

    for task in data.puzzles.iter() {
        if task.puzzle_type == puzzle_type {
            let sol = &data.solutions[&task.id];
            eprintln!(
                "{}: Len = {}. Colors = {}",
                task.id,
                sol.len(),
                task.num_colors
            );
            // check_solution(tasks, sol, &data.puzzle_info);
            new_solve(task, &data.puzzle_info);
            break;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CellType {
    Central,
    Mid,
    Corner,
}

#[derive(Clone, Debug)]
struct Block {
    cells: Vec<usize>,
    score: usize,
}

impl Block {
    pub fn new(cells: &[usize], score: usize) -> Self {
        Self {
            cells: cells.to_vec(),
            score,
        }
    }
}

fn new_solve(task: &Puzzle, puzzle_info: &BTreeMap<String, PuzzleType>) {
    let n = task.solution_state.len();
    let puzzle_info = &puzzle_info[&task.puzzle_type];
    let mut rng = rand::thread_rng();

    let blocks = get_blocks(
        task,
        &puzzle_info.moves.values().cloned().collect::<Vec<_>>(),
    );

    for (k, mov) in puzzle_info.moves.iter() {
        eprintln!("{}: {:?}", k, mov.cycles);
    }
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

    // for i in 0..move_groups.len() {
    //     let is_possible = possible_to_solve(
    //         &move_groups[i],
    //         &blocks,
    //         &task.initial_state,
    //         &task.solution_state,
    //     );
    //     eprintln!("{i}. IS POSSIBLE: {is_possible}");
    // }
    for (it, w) in move_groups.windows(2).enumerate() {
        eprintln!("{}..{}", it, it + 1);
        between_groups(&w[0], &w[1], &blocks, n);
        // if it == 2 {
        //     break;
        // }
    }
}

fn get_all_perms(a: &[usize]) -> Vec<Vec<usize>> {
    let mut res = vec![];
    if a.len() == 1 {
        res.push(a.to_vec());
    } else if a.len() == 2 {
        res.push(a.to_vec());
        res.push(vec![a[1], a[0]]);
    } else if a.len() == 3 {
        res.push(a.to_vec());
        res.push(vec![a[1], a[2], a[0]]);
        res.push(vec![a[2], a[0], a[1]]);
        res.push(vec![a[2], a[1], a[0]]);
        res.push(vec![a[0], a[2], a[1]]);
        res.push(vec![a[1], a[0], a[2]]);
    } else {
        panic!();
    }
    res
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
            eprintln!("GROUP: {:?}", cur_group);
            cnt_groups += 1;
        }
    }
    res
}

fn between_groups(
    moves1: &[SeveralMoves],
    moves2: &[SeveralMoves],
    blocks: &[Vec<usize>],
    n: usize,
) {
    let groups = get_groups(blocks, moves2);

    let calc_hash = |a: &[usize]| {
        let mut hasher = DefaultHasher::new();
        let mut inv = vec![0; n];
        for (i, &x) in a.iter().enumerate() {
            inv[x] = i;
        }
        let mut ids = vec![];
        for block in blocks.iter() {
            let mut group = vec![];
            for &x in block.iter() {
                group.push(a[x]);
            }
            let group_id = *groups.get(&group).unwrap();
            ids.push(group_id);
            hasher.write_usize(group_id);
        }
        let h = hasher.finish();
        // eprintln!("CURRENT IDS: {:?}. Hash: {h}", ids);
        h
    };
    let start_state: Vec<usize> = (0..n).collect();
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();
    seen.insert(calc_hash(&start_state));
    queue.push_back(start_state.clone());
    while let Some(state) = queue.pop_front() {
        for mov in moves1.iter() {
            let mut new_state = state.clone();
            mov.permutation.apply(&mut new_state);
            let new_hash = calc_hash(&new_state);
            if !seen.contains(&new_hash) {
                seen.insert(new_hash);
                queue.push_back(new_state);
                if queue.len() % 10000 == 0 {
                    eprintln!("Seen: {}. Queue: {}", seen.len(), queue.len());
                }
            }
        }
    }
    eprintln!("Seen: {}", seen.len());
}

fn possible_to_solve(
    available_moves: &[SeveralMoves],
    blocks: &[Vec<usize>],
    state: &[usize],
    target_state: &[usize],
) -> bool {
    let mut seen = HashSet::new();
    for block in blocks.iter() {
        let start = block.clone();
        let mut queue = VecDeque::new();
        let seen_before = seen.len();
        if !seen.insert(start.clone()) {
            continue;
        }
        queue.push_back(start);
        while let Some(positions) = queue.pop_front() {
            for mov in available_moves.iter() {
                let mut new_positions = positions.clone();
                for x in new_positions.iter_mut() {
                    *x = mov.permutation.next(*x);
                }
                if !seen.contains(&new_positions) {
                    seen.insert(new_positions.clone());
                    queue.push_back(new_positions);
                }
            }
        }
        eprintln!("Block: {:?} -> {}", block, seen.len() - seen_before);
    }
    true
}

fn create_moves(puzzle_info: &PuzzleType, moves: &[&str]) -> Vec<SeveralMoves> {
    let mut res = vec![];
    for &s in moves.iter() {
        match s.strip_suffix("x2") {
            Some(s) => {
                res.push(SeveralMoves {
                    name: vec![s.to_string(), s.to_string()],
                    permutation: puzzle_info.moves[s].x2(),
                });
                let s = rev(s);
                res.push(SeveralMoves {
                    name: vec![s.to_string(), s.to_string()],
                    permutation: puzzle_info.moves[&s].x2(),
                });
            }
            None => {
                res.push(SeveralMoves {
                    name: vec![s.to_string()],
                    permutation: puzzle_info.moves[s].clone(),
                });
                let s = rev(s);
                res.push(SeveralMoves {
                    name: vec![s.to_string()],
                    permutation: puzzle_info.moves[&s].clone(),
                });
            }
        }
    }
    res
}

struct SeveralMoves {
    name: Vec<String>,
    permutation: Permutation,
}

fn new_solve123(
    task: &Puzzle,
    mut queue: Vec<Permutation>,
    puzzle_info: &BTreeMap<String, PuzzleType>,
) {
    let mut state = task.initial_state.clone();
    println!("TASK ID: {}", task.id);
    println!("START: {}", task.convert_state(&task.solution_state));

    let puzzle_info = &puzzle_info[&task.puzzle_type];
    let mut cnt_used = vec![0; puzzle_info.n];
    for mov in puzzle_info.moves.values() {
        eprintln!("MOVE: {:?}", mov.cycles);
        for cycle in mov.cycles.iter() {
            for &x in cycle.iter() {
                cnt_used[x] += 1;
            }
        }
    }
    println!("Used: {:?}", cnt_used);
    let mut types = vec![CellType::Corner; state.len()];
    for i in 0..types.len() {
        if cnt_used[i] == 4 {
            types[i] = CellType::Central;
            for mov in puzzle_info.moves.values() {
                let mut contains = false;
                for cycle in mov.cycles.iter() {
                    if cycle.contains(&i) {
                        contains = true;
                        break;
                    }
                }
                if contains {
                    for cycle in mov.cycles.iter() {
                        for &x in cycle.iter() {
                            if types[x] == CellType::Corner {
                                types[x] = CellType::Mid;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut cnt_mid = 0;
    let mut cnt_corners = 0;
    let mut cnt_central = 0;
    for i in 0..types.len() {
        println!("{}: {:?}", i, types[i]);
        match types[i] {
            CellType::Central => cnt_central += 1,
            CellType::Mid => cnt_mid += 1,
            CellType::Corner => cnt_corners += 1,
        }
    }
    println!("Types: {} {} {}", cnt_central, cnt_mid, cnt_corners);

    let mut rng = rand::thread_rng();

    let mut blocks = vec![];
    let mut magic = vec![0; state.len()];
    for mov in puzzle_info.moves.values() {
        let xor: u64 = rng.gen();
        for cycle in mov.cycles.iter() {
            for &x in cycle.iter() {
                magic[x] ^= xor;
            }
        }
    }
    let mut seen = vec![false; state.len()];
    for i in 0..magic.len() {
        if seen[i] {
            continue;
        }
        let mut group = vec![];
        for j in 0..magic.len() {
            if magic[j] == magic[i] {
                group.push(j);
                seen[j] = true;
            }
        }
        let mut score = 0;
        if group.len() == 3 {
            score = 1;
        } else if group.len() == 2 && types[group[0]] == CellType::Mid {
            score = 100;
        } else {
            assert_eq!(types[group[0]], CellType::Central);
            score = 1000;
        }
        blocks.push(Block::new(&group, score));
    }

    for b in blocks.iter() {
        println!("Block: {:?}", b);
    }

    let calc_score = |s: &[usize]| {
        let t = &task.solution_state;
        let mut res = 0;
        for block in blocks.iter() {
            let mut ok = true;
            for &x in block.cells.iter() {
                if s[x] != t[x] {
                    ok = false;
                }
            }
            if !ok {
                res += block.score;
            }
        }
        res
    };

    let mut heap = BinaryHeap::new();
    let start_state = State {
        priority: 0,
        score: calc_score(&state),
        len: 0,
        state: state.clone(),
    };
    heap.push(Reverse(start_state));
    let mut seen = HashMap::new();
    seen.insert(state, 0);
    let mut smallest_score = usize::MAX;
    while let Some(Reverse(state)) = heap.pop() {
        if state.score < smallest_score {
            smallest_score = state.score;
            eprintln!(
                "Len={}, score={}, state={}",
                state.len,
                state.score,
                task.convert_state(&state.state)
            );
        }
        if state.score == 0 {
            println!("SOLVED!");
            break;
        }
        for mov in queue.iter() {
            let mut new_state = state.state.clone();
            mov.apply(&mut new_state);
            let new_score = calc_score(&new_state);
            let new_len = state.len + 1;
            let new_state = State {
                priority: new_score + new_len * 30,
                score: new_score,
                len: new_len,
                state: new_state,
            };
            if !seen.contains_key(&new_state.state) || seen[&new_state.state] > new_len {
                seen.insert(new_state.state.clone(), new_len);
                heap.push(Reverse(new_state));
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct State {
    priority: usize,
    score: usize,
    len: usize,
    state: Vec<usize>,
}

fn possible_positions(block: &[usize], moves: &[Permutation]) -> Vec<Vec<usize>> {
    let start = block.to_vec();
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();
    seen.insert(start.clone());
    queue.push_back(start);
    while let Some(positions) = queue.pop_front() {
        for mov in moves.iter() {
            let mut new_positions = positions.clone();
            for x in new_positions.iter_mut() {
                *x = mov.next(*x);
            }
            if !seen.contains(&new_positions) {
                seen.insert(new_positions.clone());
                queue.push_back(new_positions);
            }
        }
    }
    seen.into_iter().collect()
}

fn analyze_permuations(data: &Data) {
    for task in data.puzzles.iter() {
        eprintln!(
            "TASK ID: {}. Type: {}. Colors: {}",
            task.id, task.puzzle_type, task.num_colors
        );

        let puzzle_info = &data.puzzle_info[&task.puzzle_type];
        let moves = puzzle_info.moves.values().cloned().collect::<Vec<_>>();
        let blocks = get_blocks(task, &moves);
        let mut not_uniq = 0;
        for block in blocks.iter() {
            let positions = possible_positions(block, &moves);
            let mut cnt_ok = 0;
            for pos in positions.iter() {
                let mut ok = true;
                for i in 0..block.len() {
                    let color = task.initial_state[block[i]];
                    if task.initial_state[pos[i]] != color {
                        ok = false;
                    }
                }
                if ok {
                    cnt_ok += 1;
                }
            }
            assert!(cnt_ok > 0);
            if cnt_ok != 1 {
                not_uniq += 1;
            }
        }
        if not_uniq != 0 {
            eprintln!("Not uniq: {}", not_uniq);
        }
    }
}

fn main() {
    println!("Hello, world!");

    let data = load_data();

    // analyze_puzzle_type(&data, "cube_3/3/3");
    // analyze_permuations(&data);

    // show_info(&data);
    solve(&data, 30);
}
