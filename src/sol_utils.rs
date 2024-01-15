use crate::{data::Data, puzzle::Puzzle, utils::get_start_permutation};

#[derive(Clone)]
pub struct TaskSolution {
    pub task_id: usize,
    pub answer: Vec<String>,
    pub failed_on_stage: Option<usize>,
    pub state: Vec<usize>,
    pub task: Puzzle,
}

impl TaskSolution {
    pub fn new(data: &Data, task_id: usize) -> Self {
        TaskSolution {
            task_id,
            answer: vec![],
            failed_on_stage: None,
            state: get_start_permutation(&data.puzzles[task_id], &data.solutions.sample[&task_id]),
            task: data.puzzles[task_id].clone(),
        }
    }

    pub fn new_fake(state: Vec<usize>, task: Puzzle) -> Self {
        TaskSolution {
            task_id: 0,
            answer: vec![],
            failed_on_stage: None,
            state,
            task,
        }
    }

    pub fn all_by_type(data: &Data, task_type: &str, only_perm: bool) -> Vec<Self> {
        let mut solutions = vec![];
        for task in data.puzzles.iter() {
            if task.puzzle_type == task_type {
                if only_perm && task.color_names.len() == 6 {
                    continue;
                }
                solutions.push(TaskSolution::new(data, task.id))
            }
        }
        assert!(!solutions.is_empty());
        solutions
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
            .filter(|&i| self.task.solution_state[self.state[i]] == self.task.solution_state[i])
            .collect()
    }

    pub fn get_correct_positions(&self) -> Vec<usize> {
        (0..self.state.len())
            .filter(|&i| self.state[i] == i)
            .collect()
    }
}
