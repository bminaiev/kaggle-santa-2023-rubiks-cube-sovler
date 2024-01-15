use std::collections::{BinaryHeap, HashMap};

use crate::{
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    puzzle_type::PuzzleType,
    utils::perm_parity,
};

#[derive(Clone, Debug)]
pub struct Triangle {
    pub mv: SeveralMoves,
    pub info: Vec<String>,
}

impl Triangle {
    pub fn cycle(&self) -> [usize; 3] {
        let cycle = &self.mv.permutation.cycles[0];
        [cycle[0], cycle[1], cycle[2]]
    }

    pub fn key(&self) -> String {
        format!("{}_{}", self.info[0], self.info[2])
    }

    pub fn can_combine(&self, other: &Self) -> bool {
        if self.info[0] != other.info[0]
            || self.info[2] != other.info[2]
            || self.info[1] == other.info[1]
            || self.info[1] == rev_move(&other.info[1])
        {
            return false;
        }

        // let expected_perm = self.mv.permutation.combine(&other.mv.permutation);

        // let mut state: Vec<_> = (0..puzzle_info.n).collect();
        // let moves = Self::gen_combination_moves(&[self, other]);
        // for mv in moves.iter() {
        //     puzzle_info.moves[mv].apply(&mut state);
        // }
        // expected_perm.apply_rev(&mut state);
        // for (i, val) in state.iter().enumerate() {
        //     if *val != i {
        //         eprintln!("Bad triangles!: {:?} and {:?}", self.info, other.info);
        //         assert!(false);
        //         return false;
        //     }
        // }

        // eprintln!("Can join triangles!: {:?} and {:?}", self.info, other.info);

        true
    }

    pub fn gen_combination_moves(triangles: &[&Triangle]) -> Vec<String> {
        let mv1 = triangles[0].info[0].clone();
        let side_mv = triangles[0].info[2].clone();
        let mut res = vec![mv1.clone(), side_mv.clone()];
        for tr in triangles.iter() {
            res.push(tr.info[1].clone());
        }
        res.push(rev_move(&side_mv));
        res.push(rev_move(&mv1));
        res.push(side_mv.clone());
        for tr in triangles.iter().rev() {
            res.push(rev_move(&tr.info[1]));
        }
        res.push(rev_move(&side_mv));
        res
    }

    pub fn create(puzzle_info: &PuzzleType, mv1: &str, mv2: &str, side_mv: &str) -> Option<Self> {
        let check = [
            mv1,
            side_mv,
            mv2,
            &rev_move(side_mv),
            &rev_move(mv1),
            side_mv,
            &rev_move(mv2),
            &rev_move(side_mv),
        ];
        let mut perm = Permutation::identity();
        for mv in check.iter() {
            let check_perm = perm.combine_linear(&puzzle_info.moves[&mv.to_string()]);
            perm = check_perm;
        }
        if perm.cycles.len() == 1 {
            Some(Self {
                mv: SeveralMoves {
                    name: check.iter().map(|&x| x.to_string()).collect(),
                    permutation: perm,
                },
                info: vec![mv1.to_string(), mv2.to_string(), side_mv.to_string()],
            })
        } else {
            None
        }
    }
}

pub struct TriangleGroupSolver {
    ids: Vec<usize>,
    d: Vec<Vec<usize>>,
    triangles: Vec<Triangle>,
    pub cur_answer_len: usize,
}

impl TriangleGroupSolver {
    pub fn new(triangles: &[Triangle], state: &[usize]) -> Self {
        let mut ids = vec![];
        for tr in triangles.iter() {
            for v in tr.cycle() {
                ids.push(v);
            }
        }
        ids.sort();
        ids.dedup();

        let size = ids.len();
        let mut d = vec![vec![usize::MAX / 10; size]; size];
        for i in 0..d.len() {
            d[i][i] = 0;
        }

        let mut res = Self {
            ids,
            d,
            triangles: vec![],
            cur_answer_len: usize::MAX,
        };

        let triangles: Vec<_> = triangles
            .iter()
            .map(|tr| {
                let cycle = tr.cycle();
                let mut new_cycle = vec![0; 3];
                for i in 0..3 {
                    new_cycle[i] = res.conv_id(cycle[i]);
                }
                Triangle {
                    mv: SeveralMoves {
                        name: tr.mv.name.clone(),
                        permutation: Permutation {
                            cycles: vec![new_cycle],
                        },
                    },
                    info: tr.info.clone(),
                }
            })
            .collect();

        res.triangles = triangles;

        for tr in res.triangles.iter() {
            let cycle = tr.cycle();
            for i in 0..3 {
                let v = cycle[i];
                let u = cycle[(i + 1) % 3];
                res.d[v][u] = 1;
            }
        }
        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    res.d[i][j] = res.d[i][j].min(res.d[i][k] + res.d[k][j]);
                }
            }
        }

        res.cur_answer_len = res.solve(state).len();

        res
    }

    pub fn conv_id(&self, x: usize) -> usize {
        self.ids.binary_search(&x).unwrap()
    }

    pub fn score(&self, perm: &[usize]) -> usize {
        let mut res = 0;
        for i in 0..self.ids.len() {
            let value = perm[i];
            res += self.d[value][i];
        }
        res
    }

    fn conv_perm(&self, state: &[usize]) -> Vec<usize> {
        self.ids.iter().map(|&x| self.conv_id(state[x])).collect()
    }

    pub fn solve(&self, state: &[usize]) -> Vec<usize> {
        let perm = self.conv_perm(state);
        assert!(perm_parity(&perm) == 0);
        self.a_star(&perm)
    }

    fn a_star(&self, start: &[usize]) -> Vec<usize> {
        let mut queue = BinaryHeap::new();
        queue.push(State::new(start.to_vec(), 0, self.score(start)));
        let mut seen = HashMap::new();
        seen.insert(
            start.to_vec(),
            Prev {
                perm: start.to_vec(),
                edge_id: usize::MAX,
            },
        );
        while let Some(state) = queue.pop() {
            if state.score == 0 {
                // eprintln!("Found solution: {:?}", state.spent_moves);
                let mut res = vec![];
                let mut cur = state.perm.clone();
                while cur != start {
                    let prev = seen[&cur].clone();
                    res.push(prev.edge_id);
                    cur = prev.perm;
                }
                res.reverse();
                return res;
            }
            for (tr_id, tr) in self.triangles.iter().enumerate() {
                let mut new_perm = state.perm.clone();
                tr.mv.permutation.apply(&mut new_perm);
                let new_score = self.score(&new_perm);
                if seen.contains_key(&new_perm) {
                    continue;
                }
                seen.insert(
                    new_perm.clone(),
                    Prev {
                        perm: state.perm.clone(),
                        edge_id: tr_id,
                    },
                );
                queue.push(State::new(new_perm, state.spent_moves + 1, new_score));
            }
        }
        unreachable!();
    }

    pub(crate) fn get_dist_estimate(&self, state: &[usize]) -> usize {
        let perm = self.conv_perm(state);
        self.score(&perm)
    }
}

#[derive(Clone)]
struct Prev {
    perm: Vec<usize>,
    edge_id: usize,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct State {
    priority: usize,
    perm: Vec<usize>,
    spent_moves: usize,
    score: usize,
}

impl State {
    pub fn new(perm: Vec<usize>, spent_moves: usize, score: usize) -> Self {
        Self {
            priority: usize::MAX - (spent_moves + score),
            perm,
            spent_moves,
            score,
        }
    }
}
