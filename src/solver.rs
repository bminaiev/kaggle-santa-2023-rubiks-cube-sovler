use crate::{data::Data, puzzle::Puzzle, utils::get_start_permutation};

pub fn solve(data: &Data, task_id: usize) {
    eprintln!("SOLVING {task_id}");
    let task = &data.puzzles[task_id];
    let example_solution = &data.solutions[&task_id];
    let start_state = get_start_permutation(task, example_solution);
    eprintln!("START STATE: {:?}", start_state);
}
