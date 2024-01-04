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
    moves::SeveralMoves,
    permutation::Permutation,
    puzzle::Puzzle,
    puzzle_type::PuzzleType,
    solver::solve,
    utils::{get_all_perms, get_blocks},
};

pub mod checker;
pub mod data;
pub mod moves;
pub mod permutation;
pub mod puzzle;
pub mod puzzle_type;
pub mod solver;
pub mod utils;

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
    solve(&data, 31);
}
