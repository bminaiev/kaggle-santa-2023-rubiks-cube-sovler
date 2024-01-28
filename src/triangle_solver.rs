use std::{collections::HashMap, mem};

use crate::{
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    puzzle_type::PuzzleType,
    utils::perm_parity,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Triangle {
    pub mv: SeveralMoves,
    pub mv1: String,
    pub mv2: String,
    pub side_mv: String,
}

fn mv_key(mv: &str) -> String {
    if mv.starts_with('-') {
        mv[1..2].to_string()
    } else {
        mv[..1].to_string()
    }
}

impl Triangle {
    pub fn cycle(&self) -> [usize; 3] {
        let cycle = &self.mv.permutation.cycles[0];
        [cycle[0], cycle[1], cycle[2]]
    }

    pub fn key(&self) -> String {
        format!(
            "{}_{}_{}",
            self.side_mv,
            mv_key(&self.mv1),
            mv_key(&self.mv2)
        )
        // self.side_mv.to_string()
    }

    pub fn gen_combination_moves(
        moves1: &[String],
        moves2: &[String],
        side_mv: &str,
    ) -> Vec<String> {
        let mut res = vec![];
        for mv1 in moves1.iter() {
            res.push(mv1.clone());
        }
        res.push(side_mv.to_string());
        for mv2 in moves2.iter() {
            res.push(mv2.clone());
        }
        res.push(rev_move(side_mv));
        for mv1 in moves1.iter() {
            res.push(rev_move(mv1));
        }
        res.push(side_mv.to_string());
        for mv2 in moves2.iter() {
            res.push(rev_move(mv2));
        }
        res.push(rev_move(side_mv));
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
                mv1: mv1.to_string(),
                mv2: mv2.to_string(),
                side_mv: side_mv.to_string(),
            })
        } else {
            None
        }
    }
}

pub struct TriangleGroupSolver {
    positions: Vec<usize>,
    colors: Vec<usize>,
    d: Vec<Vec<usize>>,
    triangles: Vec<Triangle>,
    pub cur_answer_len: usize,
}

pub enum Solver {
    Astar,
    Bfs(usize),
}

impl Default for Solver {
    fn default() -> Self {
        Solver::Bfs(20)
    }
}

impl TriangleGroupSolver {
    pub fn new(triangles: &[Triangle], start_colors: &[usize], target_colors: &[usize]) -> Self {
        let mut positions = vec![];
        for tr in triangles.iter() {
            for v in tr.cycle() {
                positions.push(v);
            }
        }
        positions.sort();
        positions.dedup();
        let mut colors: Vec<_> = positions.iter().map(|&x| start_colors[x]).collect();
        colors.sort();
        colors.dedup();

        let d = vec![vec![usize::MAX / 10; colors.len()]; positions.len()];

        let mut res = Self {
            positions,
            colors,
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
                    new_cycle[i] = res.conv_pos(cycle[i]);
                }
                Triangle {
                    mv: SeveralMoves {
                        name: tr.mv.name.clone(),
                        permutation: Permutation {
                            cycles: vec![new_cycle],
                        },
                    },
                    mv1: tr.mv1.clone(),
                    mv2: tr.mv2.clone(),
                    side_mv: tr.side_mv.clone(),
                }
            })
            .collect();

        res.triangles = triangles;

        for i in 0..res.positions.len() {
            let target_color = res.conv_color(target_colors[res.positions[i]]);
            res.d[i][target_color] = 0;
        }
        loop {
            let mut changed = false;
            for tr in res.triangles.iter() {
                let cycle = tr.cycle();
                for i in 0..3 {
                    let v = cycle[i];
                    let u = cycle[(i + 1) % 3];
                    for c in 0..res.colors.len() {
                        if res.d[v][c] > 1 + res.d[u][c] {
                            res.d[v][c] = 1 + res.d[u][c];
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        res.cur_answer_len = res.solve(start_colors, Solver::default()).len();

        res
    }

    pub fn conv_color(&self, x: usize) -> usize {
        self.colors.binary_search(&x).unwrap()
    }

    pub fn conv_pos(&self, x: usize) -> usize {
        self.positions.binary_search(&x).unwrap()
    }

    fn score(&self, perm: ColorsState) -> usize {
        let mut res = 0;
        for i in 0..self.positions.len() {
            let value = perm.get(i);
            res += self.d[i][value];
        }
        res
    }

    fn conv_state(&self, state: &[usize]) -> ColorsState {
        let mut res = ColorsState::default();
        for i in 0..self.positions.len() {
            let color = self.conv_color(state[self.positions[i]]);
            res.set(i, color);
        }
        res
    }

    pub fn solve(&self, state: &[usize], solver: Solver) -> Vec<usize> {
        let perm = self.conv_state(state);
        if self.colors.len() == 24 {
            let mut expected_colors = vec![usize::MAX; self.colors.len()];
            for i in 0..expected_colors.len() {
                for c in 0..24 {
                    if self.d[i][c] == 0 {
                        expected_colors[i] = c;
                    }
                }
            }
            let my = perm.conv();
            let parity1 = perm_parity(&my);
            let parity2 = perm_parity(&expected_colors);
            assert_eq!(parity1, parity2);
        }
        // assert!(perm_parity(&perm) == 0);
        match solver {
            Solver::Astar => unimplemented!(),
            Solver::Bfs(width) => self.bfs(perm, width),
        }
    }

    // fn a_star_old(&self, start: &[usize]) -> Vec<usize> {
    //     let start = ColorsState::from_array(start);
    //     let mut queue = BinaryHeap::new();
    //     queue.push(State::new(start, 0, self.score(start)));
    //     let mut seen = HashMap::new();
    //     seen.insert(
    //         start.to_vec(),
    //         Prev {
    //             perm: ColorsState::from_array(start),
    //             edge_id: usize::MAX,
    //         },
    //     );
    //     while let Some(state) = queue.pop() {
    //         if state.score == 0 {
    //             // eprintln!("Found solution: {:?}", state.spent_moves);
    //             let mut res = vec![];
    //             let mut cur = state.perm.clone();
    //             while cur != start {
    //                 let prev = seen[&cur].clone();
    //                 res.push(prev.edge_id);
    //                 cur = prev.perm;
    //             }
    //             res.reverse();
    //             return res;
    //         }
    //         for (tr_id, tr) in self.triangles.iter().enumerate() {
    //             let mut new_perm = state.perm.clone();
    //             tr.mv.permutation.apply(&mut new_perm);
    //             let new_score = self.score(&new_perm);
    //             if seen.contains_key(&new_perm) {
    //                 continue;
    //             }
    //             seen.insert(
    //                 new_perm.clone(),
    //                 Prev {
    //                     perm: state.perm.clone(),
    //                     edge_id: tr_id,
    //                 },
    //             );
    //             queue.push(State::new(new_perm, state.spent_moves + 1, new_score));
    //         }
    //     }
    //     unreachable!();
    // }

    fn bfs(&self, start: ColorsState, width: usize) -> Vec<usize> {
        let mut queues = vec![vec![]; 50];
        queues[0].push(State::new(start, 0, self.score(start)));
        let mut seen = HashMap::new();
        seen.insert(
            start,
            Prev {
                perm: start,
                edge_id: usize::MAX,
            },
        );
        // const MAX_SEE: usize = 5;
        for it in 0..queues.len() {
            let mut queue = vec![];
            mem::swap(&mut queue, &mut queues[it]);
            queue.sort_by_key(|x| x.score);
            queue.truncate(width);
            for state in queue.into_iter() {
                if state.score == 0 {
                    // eprintln!("Found solution: {:?}", state.spent_moves);
                    let mut res = vec![];
                    let mut cur = state.perm;
                    while cur != start {
                        let prev = seen[&cur].clone();
                        res.push(prev.edge_id);
                        cur = prev.perm;
                    }
                    res.reverse();
                    return res;
                }
                for (tr_id, tr) in self.triangles.iter().enumerate() {
                    let mut new_perm = state.perm;
                    new_perm.apply_perm(&tr.mv.permutation);
                    let new_score = self.score(new_perm);
                    if seen.contains_key(&new_perm) {
                        continue;
                    }
                    seen.insert(
                        new_perm,
                        Prev {
                            perm: state.perm,
                            edge_id: tr_id,
                        },
                    );
                    queues[it + 1].push(State::new(new_perm, state.spent_moves + 1, new_score));
                }
            }
        }
        unreachable!();
    }

    pub(crate) fn get_dist_estimate(&self, state: &[usize]) -> usize {
        let perm = self.conv_state(state);
        self.score(perm)
    }
}

#[derive(Clone)]
struct Prev {
    perm: ColorsState,
    edge_id: usize,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct State {
    priority: usize,
    perm: ColorsState,
    spent_moves: usize,
    score: usize,
}

impl State {
    pub fn new(perm: ColorsState, spent_moves: usize, score: usize) -> Self {
        Self {
            priority: usize::MAX - (spent_moves + score),
            perm,
            spent_moves,
            score,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
struct ColorsState {
    mask: u128,
}

const BITS_PER_COLOR: usize = 5;

impl ColorsState {
    fn get(&self, pos: usize) -> usize {
        ((self.mask >> (pos * BITS_PER_COLOR)) & ((1 << BITS_PER_COLOR) - 1)) as usize
    }

    fn set(&mut self, pos: usize, color: usize) {
        assert!(color < (1 << BITS_PER_COLOR));
        self.mask &= !(((1 << BITS_PER_COLOR) - 1) << (pos * BITS_PER_COLOR));
        self.mask |= (color as u128) << (pos * BITS_PER_COLOR);
    }

    fn swap(&mut self, p1: usize, p2: usize) {
        let c1 = self.get(p1);
        let c2 = self.get(p2);
        self.set(p1, c2);
        self.set(p2, c1);
    }

    fn apply_perm(&mut self, perm: &Permutation) {
        assert_eq!(perm.cycles.len(), 1);
        for cycle in perm.cycles.iter() {
            for w in cycle.windows(2) {
                self.swap(w[0], w[1]);
            }
        }
    }

    fn conv(&self) -> Vec<usize> {
        let mut res = vec![];
        for i in 0..24 {
            res.push(self.get(i));
        }
        res
    }
}
