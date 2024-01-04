use std::collections::BTreeMap;

use crate::permutation::Permutation;

#[derive(Debug, Clone)]
pub struct PuzzleType {
    pub n: usize,
    pub moves: BTreeMap<String, Permutation>,
}
