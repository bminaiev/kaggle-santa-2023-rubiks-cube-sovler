use crate::{data::Data, utils::get_start_permutation};

#[derive(Clone)]
pub struct TaskSolution {
    pub task_id: usize,
    pub answer: Vec<String>,
    pub failed_on_stage: Option<usize>,
    pub state: Vec<usize>,
}

impl TaskSolution {
    pub fn new(data: &Data, task_id: usize) -> Self {
        TaskSolution {
            task_id,
            answer: vec![],
            failed_on_stage: None,
            state: get_start_permutation(&data.puzzles[task_id], &data.solutions[&task_id]),
        }
    }

    pub fn all_by_type(data: &Data, task_type: &str) -> Vec<Self> {
        let mut solutions = vec![];
        for task in data.puzzles.iter() {
            if task.puzzle_type == task_type {
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
}
