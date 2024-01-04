use crate::puzzle_type::PuzzleType;

pub struct Puzzle {
    pub id: usize,
    pub puzzle_type: String,
    pub solution_state: Vec<usize>,
    pub initial_state: Vec<usize>,
    pub num_wildcards: usize,
    pub num_colors: usize,
    pub color_names: Vec<String>,
    pub info: PuzzleType,
}

impl Puzzle {
    pub fn convert_state(&self, state: &[usize]) -> String {
        state
            .iter()
            .map(|x| self.color_names[*x].clone())
            .collect::<Vec<_>>()
            .join(";")
    }
}
