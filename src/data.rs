use std::collections::{BTreeMap, HashMap};

use crate::{permutation::Permutation, puzzle::Puzzle, puzzle_type::PuzzleType};

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

fn load_puzzles(puzzle_info: &BTreeMap<String, PuzzleType>) -> Vec<Puzzle> {
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
        let mut color_names = vec![String::new(); colors.len()];
        for (k, v) in colors.iter() {
            color_names[*v] = k.clone();
        }

        puzzles.push(Puzzle {
            id,
            solution_state,
            initial_state,
            num_wildcards,
            num_colors: colors.len(),
            color_names,
            info: puzzle_info.get(&puzzle_type).unwrap().clone(),
            puzzle_type,
        });
    }

    puzzles
}

pub struct Solutions {
    pub sample: BTreeMap<usize, Vec<String>>,
    // https://www.kaggle.com/code/seanbearden/solve-all-nxnxn-cubes-w-traditional-solution-state/output
    pub s853k: BTreeMap<usize, Vec<String>>,
    pub my_280k: BTreeMap<usize, Vec<String>>,
    pub ab: BTreeMap<usize, Vec<String>>,
}

fn load_solutions_file(filename: &str) -> BTreeMap<usize, Vec<String>> {
    let mut reader = csv::Reader::from_path(format!("data/{filename}")).unwrap();

    let mut solutions: BTreeMap<usize, Vec<String>> = BTreeMap::new();

    for result in reader.records() {
        let record = result.unwrap();
        let id = record[0].parse::<usize>().unwrap();
        let solution: Vec<String> = record[1].split('.').map(|s| s.to_string()).collect();

        solutions.insert(id, solution);
    }

    solutions
}

fn load_solutions() -> Solutions {
    Solutions {
        sample: load_solutions_file("sample_submission.csv"),
        s853k: load_solutions_file("submission_853k.csv"),
        my_280k: load_solutions_file("my_280k.csv"),
        ab: load_solutions_file("../dwalton76/sub_ab.csv"),
    }
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

pub struct Data {
    pub puzzle_info: BTreeMap<String, PuzzleType>,
    pub puzzles: Vec<Puzzle>,
    pub solutions: Solutions,
}

pub fn load_data() -> Data {
    let puzzle_info = load_puzzle_info();
    let puzzles = load_puzzles(&puzzle_info);
    let solutions = load_solutions();
    Data {
        puzzle_info,
        puzzles,
        solutions,
    }
}
