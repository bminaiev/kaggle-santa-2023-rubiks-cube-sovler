use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap, HashSet},
    time::Instant,
};

use rand::{seq::SliceRandom, Rng};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Permutation {
    cycles: Vec<Vec<usize>>,
}

impl Permutation {
    pub fn identity() -> Self {
        Self { cycles: vec![] }
    }

    pub fn sum_len(&self) -> usize {
        self.cycles.iter().map(|c| c.len()).sum()
    }

    pub fn from_array(p: &[usize]) -> Self {
        let mut cycles = vec![];
        let mut seen = vec![false; p.len()];
        for start in 0..p.len() {
            let mut cycle = vec![];
            let mut cur = start;
            while !seen[cur] {
                cycle.push(cur);
                seen[cur] = true;
                cur = p[cur];
            }
            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }
        Self { cycles }
    }

    pub fn apply(&self, a: &mut [usize]) {
        for cycle in self.cycles.iter() {
            for w in cycle.windows(2) {
                a.swap(w[0], w[1]);
            }
        }
    }

    pub fn combine(&self, other: &Self) -> Self {
        let mut a = HashMap::new();
        for who in [self, other].iter() {
            for cycle in who.cycles.iter() {
                for w in cycle.windows(2) {
                    let v1 = *a.get(&w[0]).unwrap_or(&w[0]);
                    let v2 = *a.get(&w[1]).unwrap_or(&w[1]);
                    a.insert(w[0], v2);
                    a.insert(w[1], v1);
                }
            }
        }
        let mut cycles = vec![];
        let mut seen = HashSet::new();
        let mut all_keys: Vec<usize> = a.keys().cloned().collect();
        all_keys.sort();
        for &start in all_keys.iter() {
            let mut cycle = vec![];
            let mut cur = start;
            while !seen.contains(&cur) {
                cycle.push(cur);
                seen.insert(cur);
                cur = a[&cur];
            }
            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }
        Self { cycles }
    }

    pub fn inv(&self) -> Self {
        let cycles = self
            .cycles
            .iter()
            .map(|cycle| cycle.iter().rev().cloned().collect())
            .collect();
        Self { cycles }
    }
}

#[derive(Debug)]
struct PuzzleType {
    n: usize,
    moves: BTreeMap<String, Permutation>,
}

fn load_puzzle_info() -> BTreeMap<String, PuzzleType> {
    let mut reader = csv::Reader::from_path("data/puzzle_info.csv").unwrap();

    let mut puzzle_info = BTreeMap::new();

    for result in reader.records() {
        let record = result.unwrap();
        let name = record[0].to_string();
        let moves: String = record[1].to_string();
        let moves = moves.replace('\'', "\"");

        let moves: BTreeMap<String, Vec<usize>> = serde_json::from_str(&moves).unwrap();
        let n = moves.values().next().unwrap().len();
        let moves: BTreeMap<String, Permutation> = moves
            .into_iter()
            .map(|(k, v)| (k, Permutation::from_array(&v)))
            .collect();

        let mut all_moves = moves.clone();
        for (k, v) in moves.iter() {
            all_moves.insert(format!("-{}", k), v.inv());
        }
        puzzle_info.insert(
            name,
            PuzzleType {
                n,
                moves: all_moves,
            },
        );
    }

    puzzle_info
}

struct Puzzle {
    id: usize,
    puzzle_type: String,
    solution_state: Vec<usize>,
    initial_state: Vec<usize>,
    num_wildcards: usize,
    num_colors: usize,
}

fn conv_colors(a: &[String], hm: &mut HashMap<String, usize>) -> Vec<usize> {
    let mut res = vec![];
    for x in a.iter() {
        if !hm.contains_key(x) {
            let new_id = hm.len();
            hm.insert(x.clone(), new_id);
        }
        res.push(*hm.get(x).unwrap());
    }
    res
}

fn load_puzzles() -> Vec<Puzzle> {
    let mut reader = csv::Reader::from_path("data/puzzles.csv").unwrap();

    let mut puzzles: Vec<Puzzle> = Vec::new();

    for result in reader.records() {
        let record = result.unwrap();
        let id = record[0].parse::<usize>().unwrap();
        let puzzle_type = record[1].to_string();
        let solution_state: Vec<String> = record[2].split(';').map(|s| s.to_string()).collect();
        let initial_state: Vec<String> = record[3].split(';').map(|s| s.to_string()).collect();
        let num_wildcards = record[4].parse::<usize>().unwrap();

        let mut colors = HashMap::new();
        let solution_state = conv_colors(&solution_state, &mut colors);
        let initial_state = conv_colors(&initial_state, &mut colors);

        puzzles.push(Puzzle {
            id,
            puzzle_type,
            solution_state,
            initial_state,
            num_wildcards,
            num_colors: colors.len(),
        });
    }

    puzzles
}

fn load_solutions() -> HashMap<usize, Vec<String>> {
    let mut reader = csv::Reader::from_path("data/sample_submission.csv").unwrap();

    let mut solutions: HashMap<usize, Vec<String>> = HashMap::new();

    for result in reader.records() {
        let record = result.unwrap();
        let id = record[0].parse::<usize>().unwrap();
        let solution: Vec<String> = record[1].split('.').map(|s| s.to_string()).collect();

        solutions.insert(id, solution);
    }

    solutions
}

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

struct Data {
    puzzle_info: BTreeMap<String, PuzzleType>,
    puzzles: Vec<Puzzle>,
    solutions: HashMap<usize, Vec<String>>,
}

fn load_data() -> Data {
    let puzzle_info = load_puzzle_info();
    let puzzles = load_puzzles();
    let solutions = load_solutions();
    Data {
        puzzle_info,
        puzzles,
        solutions,
    }
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
    for puzzle_id in sorted_test_ids[..10].iter() {
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
        check_solution2(puzzle, sol, &puzzle_info);
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
            new_solve(task, queue.clone());
            break;
        }
    }
}

fn new_solve(task: &Puzzle, mut queue: Vec<Permutation>) {
    let mut state = task.initial_state.clone();
    let calc_score = |s: &[usize]| {
        let t = &task.solution_state;
        let mut cnt = 0;
        for i in 0..s.len() {
            if s[i] != t[i] {
                cnt += 1;
            }
        }
        cnt as f64
    };

    let mut rng = rand::thread_rng();
    const MAX_SEC: f64 = 10.0;
    let temp_start = 10.0f64;
    let temp_end = 0.1f64;
    let start = Instant::now();
    let mut prev_score = calc_score(&state);
    let mut zz = 0;
    let mut min_score = f64::MAX;
    loop {
        zz += 1;
        if prev_score < min_score {
            min_score = prev_score;
            eprintln!("{:?}: {}", start.elapsed(), min_score);
        }
        let elapsed_s = start.elapsed().as_secs_f64();
        if elapsed_s > MAX_SEC {
            break;
        }
        let elapsed_frac = elapsed_s / MAX_SEC;
        let temp = temp_start * (temp_end / temp_start).powf(elapsed_frac);
        let m_id = rng.gen_range(0..queue.len());
        let m = &queue[m_id];

        let mut new_state = state.clone();
        m.apply(&mut new_state);

        let new_score = calc_score(&new_state);
        if new_score < prev_score || fastrand::f64() < ((prev_score - new_score) / temp).exp() {
            // Using a new state!
            prev_score = new_score;
            state = new_state;
        } else {
            // Rollback
            // perm[fr..to].reverse();
        }
    }
    let score = calc_score(&state);
    eprintln!("Score: {score}");
}

fn main() {
    println!("Hello, world!");

    let data = load_data();

    analyze_puzzle_type(&data, "globe_2/6");

    // show_info(&data);
}
