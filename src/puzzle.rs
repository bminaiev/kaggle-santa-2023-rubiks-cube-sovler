use crate::puzzle_type::PuzzleType;

#[derive(Clone)]
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

    pub fn convert_state_to_colors(&self, state: &[usize]) -> Vec<usize> {
        state.iter().map(|&x| self.solution_state[x]).collect()
    }

    pub fn convert_solution(&self, solution: &[String]) -> String {
        let mut state = self.initial_state.clone();
        for step in solution.iter() {
            let perm = &self.info.moves[step];
            for cycle in perm.cycles.iter() {
                for w in cycle.windows(2) {
                    state.swap(w[0], w[1]);
                }
            }
        }
        self.convert_state(&state)
    }

    pub fn get_color_type(&self) -> String {
        self.color_names[self.solution_state[1]].clone()
    }
}
