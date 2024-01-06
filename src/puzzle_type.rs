use std::collections::BTreeMap;

use crate::permutation::Permutation;

#[derive(Debug, Clone)]
pub struct PuzzleType {
    pub n: usize,
    pub moves: BTreeMap<String, Permutation>,
}

impl PuzzleType {
    pub fn get_all_moves(&self) -> Vec<Permutation> {
        self.moves.values().cloned().collect()
    }
}
