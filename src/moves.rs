use crate::{permutation::Permutation, puzzle_type::PuzzleType};

fn rev_move(s: &str) -> String {
    if s.starts_with('-') {
        s[1..].to_string()
    } else {
        format!("-{}", s)
    }
}

// support stuff like ["f0x2", "f1"]
pub fn create_moves(puzzle_info: &PuzzleType, moves_descr: &[&str]) -> Vec<SeveralMoves> {
    let mut res = vec![];
    for &s in moves_descr.iter() {
        match s.strip_suffix("x2") {
            Some(s) => {
                res.push(SeveralMoves {
                    name: vec![s.to_string(), s.to_string()],
                    permutation: puzzle_info.moves[s].x2(),
                });
                let s = rev_move(s);
                res.push(SeveralMoves {
                    name: vec![s.to_string(), s.to_string()],
                    permutation: puzzle_info.moves[&s].x2(),
                });
            }
            None => {
                res.push(SeveralMoves {
                    name: vec![s.to_string()],
                    permutation: puzzle_info.moves[s].clone(),
                });
                let s = rev_move(s);
                res.push(SeveralMoves {
                    name: vec![s.to_string()],
                    permutation: puzzle_info.moves[&s].clone(),
                });
            }
        }
    }
    res
}

pub struct SeveralMoves {
    pub name: Vec<String>,
    pub permutation: Permutation,
}
