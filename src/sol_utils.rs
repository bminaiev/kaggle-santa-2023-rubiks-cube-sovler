use std::collections::BTreeMap;

use crate::{data::Data, puzzle::Puzzle, utils::get_start_permutation};

#[derive(Clone)]
pub struct TaskSolution {
    pub task_id: usize,
    pub answer: Vec<String>,
    pub failed_on_stage: Option<usize>,
    // state[i] = j means that i-th cell has color j
    pub state: Vec<usize>,
    pub target_state: Vec<usize>,
    pub task: Puzzle,
    pub exact_perm: bool,
}

impl TaskSolution {
    pub fn new(data: &Data, task_id: usize) -> Self {
        let mut all_colors = vec![];
        for &color in data.puzzles[task_id].solution_state.iter() {
            if !all_colors.contains(&color) {
                all_colors.push(color);
            }
        }
        let conv = |colors: &[usize]| {
            colors
                .iter()
                .map(|&x| all_colors.iter().position(|&y| y == x).unwrap())
                .collect::<Vec<_>>()
        };

        let state = conv(&data.puzzles[task_id].initial_state);
        let target_state = conv(&data.puzzles[task_id].solution_state);

        let mut exact_perm = true;
        if data.puzzles[task_id].get_color_type() == "A" {
            exact_perm = false;
        }

        TaskSolution {
            task_id,
            answer: vec![],
            failed_on_stage: None,
            state,
            target_state,
            task: data.puzzles[task_id].clone(),
            exact_perm,
        }
    }

    pub fn reset(&mut self, data: &Data) {
        let mut all_colors = vec![];
        for &color in data.puzzles[self.task_id].solution_state.iter() {
            if !all_colors.contains(&color) {
                all_colors.push(color);
            }
        }
        let conv = |colors: &[usize]| {
            colors
                .iter()
                .map(|&x| all_colors.iter().position(|&y| y == x).unwrap())
                .collect::<Vec<_>>()
        };

        let state = conv(&data.puzzles[self.task_id].initial_state);

        self.answer.clear();
        self.failed_on_stage = None;
        self.state = state;
    }

    pub fn new_fake(state: Vec<usize>, task: Puzzle) -> Self {
        TaskSolution {
            task_id: 0,
            answer: vec![],
            failed_on_stage: None,
            target_state: (0..state.len()).collect(),
            state,
            task,
            exact_perm: true,
        }
    }

    pub fn append_move(&mut self, mv: &str) {
        self.answer.push(mv.to_owned());
        self.task.info.moves[mv].apply(&mut self.state);
    }

    pub fn all_by_type(data: &Data, task_type: &str, only_perm: bool) -> Vec<Self> {
        let mut solutions = vec![];
        for task in data.puzzles.iter() {
            if task.puzzle_type.starts_with(task_type) {
                if only_perm && task.color_names.len() == 6 {
                    continue;
                }
                solutions.push(TaskSolution::new(data, task.id))
            }
        }
        assert!(!solutions.is_empty());
        solutions
    }

    pub fn from_solutions_file(data: &Data, solutions: &BTreeMap<usize, Vec<String>>) -> Vec<Self> {
        let mut res = vec![];
        for (&task_id, sol) in solutions.iter() {
            let mut task = TaskSolution::new(data, task_id);
            for mv in sol.iter() {
                task.append_move(mv);
            }
            assert!(task.is_solved_with_wildcards());
            res.push(task);
        }
        res
    }

    pub fn print(&self, data: &Data) {
        let task = &data.puzzles[self.task_id];
        eprintln!(
            "TASK: {}. SOL LEN={}, colors = {}, State={}",
            self.task_id,
            self.answer.len(),
            task.num_colors,
            task.convert_solution(&self.answer)
        );
    }

    pub fn get_correct_colors_positions(&self) -> Vec<usize> {
        (0..self.state.len())
            .filter(|&i| self.target_state[i] == self.state[i])
            .collect()
    }
    pub(crate) fn is_solved(&self) -> bool {
        self.get_correct_colors_positions().len() == self.state.len()
    }

    pub(crate) fn is_solved_with_wildcards(&self) -> bool {
        self.get_correct_colors_positions().len() + self.task.num_wildcards >= self.state.len()
    }
}
