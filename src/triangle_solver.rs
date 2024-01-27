use std::{
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    mem,
};

use crate::{
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    puzzle_type::PuzzleType,
    utils::{perm_parity, slice_hash},
};

#[derive(Clone, Debug)]
pub struct Triangle {
    pub mv: SeveralMoves,
    mv1: String,
    mv2: String,
    side_mv: String,
}

impl Triangle {
    pub fn cycle(&self) -> [usize; 3] {
        let cycle = &self.mv.permutation.cycles[0];
        [cycle[0], cycle[1], cycle[2]]
    }

    pub fn key(&self) -> String {
        format!("{}_{}", self.mv1, self.side_mv)
    }

    pub fn can_combine(&self, other: &Self) -> bool {
        if self.mv1 != other.mv1
            || self.side_mv != other.side_mv
            || self.mv2 == other.mv2
            || self.mv2 == rev_move(&other.mv2)
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
        let mv1 = triangles[0].mv1.clone();
        let side_mv = triangles[0].side_mv.clone();
        let mut res = vec![mv1.clone(), side_mv.clone()];
        for tr in triangles.iter() {
            res.push(tr.mv2.clone());
        }
        res.push(rev_move(&side_mv));
        res.push(rev_move(&mv1));
        res.push(side_mv.clone());
        for tr in triangles.iter().rev() {
            res.push(rev_move(&tr.mv2));
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
        Solver::Bfs(5)
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

        let mut d = vec![vec![usize::MAX / 10; colors.len()]; positions.len()];

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

        eprintln!("Triangles len: {}", res.triangles.len());

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
        // {
        //     eprintln!(
        //         "Checking first moves. Cur answer len: {}",
        //         res.cur_answer_len
        //     );
        //     let start_estimate = res.get_dist_estimate(start_colors);
        //     eprintln!("Start estimate: {}", start_estimate);
        //     let start_state = res.conv_state(start_colors);
        //     for tr in res.triangles.iter() {
        //         let mut new_state = start_state.clone();
        //         tr.mv.permutation.apply(&mut new_state);
        //         let estimate = res.score(&new_state);
        //         if estimate >= start_estimate {
        //             continue;
        //         }
        //         let new_len = res.bfs(&new_state, 5).len();
        //         if new_len < res.cur_answer_len {
        //             eprintln!(
        //                 "Good first move! {} -> {}. Est: {estimate}",
        //                 res.cur_answer_len, new_len
        //             );
        //         }
        //     }
        // }
        // let bfs10 = res.solve(start_colors, Solver::Bfs(10)).len();
        // let bfs50 = res.solve(start_colors, Solver::Bfs(50)).len();
        // let bfs100 = res.solve(start_colors, Solver::Bfs(100)).len();
        // let bfs1000 = res.solve(start_colors, Solver::Bfs(1000)).len();
        // let estimate = res.get_dist_estimate(start_colors);
        // eprintln!(
        //     "BFS10: {}, BFS50: {}, BFS100: {}. A*: {}. BFS1000: {}. Esimate: {estimate}",
        //     bfs10, bfs50, bfs100, res.cur_answer_len, bfs1000
        // );

        res
    }

    pub fn conv_color(&self, x: usize) -> usize {
        self.colors.binary_search(&x).unwrap()
    }

    pub fn conv_pos(&self, x: usize) -> usize {
        self.positions.binary_search(&x).unwrap()
    }

    pub fn score(&self, perm: &[usize]) -> usize {
        let mut res = 0;
        for i in 0..self.positions.len() {
            let value = perm[i];
            res += self.d[i][value];
        }
        res
    }

    fn conv_state(&self, state: &[usize]) -> Vec<usize> {
        self.positions
            .iter()
            .map(|&x| self.conv_color(state[x]))
            .collect()
    }

    pub fn solve(&self, state: &[usize], solver: Solver) -> Vec<usize> {
        let perm = self.conv_state(state);
        // assert!(perm_parity(&perm) == 0);
        match solver {
            Solver::Astar => self.a_star_old(&perm),
            Solver::Bfs(width) => self.bfs(&perm, width),
        }
    }

    fn a_star_old(&self, start: &[usize]) -> Vec<usize> {
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

    fn bfs(&self, start: &[usize], width: usize) -> Vec<usize> {
        let mut queues = vec![vec![]; 50];
        queues[0].push(State::new(start.to_vec(), 0, self.score(start)));
        let mut seen = HashMap::new();
        seen.insert(
            start.to_vec(),
            Prev {
                perm: start.to_vec(),
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
                    queues[it + 1].push(State::new(new_perm, state.spent_moves + 1, new_score));
                }
            }
        }
        unreachable!();
    }

    pub(crate) fn get_dist_estimate(&self, state: &[usize]) -> usize {
        let perm = self.conv_state(state);
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
