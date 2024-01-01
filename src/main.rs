use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
struct Permutation {
    cycles: Vec<Vec<usize>>,
}

impl Permutation {
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
    solution_state: Vec<String>,
    initial_state: Vec<String>,
    num_wildcards: usize,
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

        puzzles.push(Puzzle {
            id,
            puzzle_type,
            solution_state,
            initial_state,
            num_wildcards,
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

fn main() {
    println!("Hello, world!");

    let puzzle_info = load_puzzle_info();
    for (k, v) in puzzle_info.iter() {
        println!("{}: n={}, cnt_moves={}", k, v.n, v.moves.len());
    }

    let puzzles = load_puzzles();
    let solutions = load_solutions();
    let mut sum_lens = 0;
    for puzzle in puzzles.iter() {
        let sol = &solutions[&puzzle.id];
        println!(
            "{}: {}. Len: {}. Sol len: {}.",
            puzzle.id,
            puzzle.puzzle_type,
            puzzle.solution_state.len(),
            sol.len()
        );
        sum_lens += sol.len();
        check_solution(puzzle, sol, &puzzle_info);
    }
    println!("Total len: {}", sum_lens);
}
