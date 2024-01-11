use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::{moves::SeveralMoves, permutation::Permutation, utils::perm_parity};

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
}

pub fn solve_triangle(start_permutation: &[usize], triangles: &[Triangle]) -> Option<Vec<usize>> {
    let mut ids = vec![];
    for tr in triangles.iter() {
        for v in tr.cycle() {
            ids.push(v);
        }
    }
    ids.sort();
    ids.dedup();

    let size = ids.len();
    // eprintln!("size={}, moves={}", ids.len(), triangles.len());
    let conv_id = |x: usize| -> usize { ids.binary_search(&x).unwrap() };
    let mut d = vec![vec![usize::MAX / 10; size]; size];
    for i in 0..d.len() {
        d[i][i] = 0;
    }
    for tr in triangles.iter() {
        let cycle = tr.cycle();
        for i in 0..3 {
            let v = conv_id(cycle[i]);
            let u = conv_id(cycle[(i + 1) % 3]);
            d[v][u] = 1;
        }
    }
    let sz = d.len();
    for i in 0..sz {
        for j in 0..sz {
            for k in 0..sz {
                d[i][j] = d[i][j].min(d[i][k] + d[k][j]);
            }
        }
    }
    // for i in 0..size {
    //     eprintln!("{:?}", d[i]);
    // }
    let score = |perm: &[usize]| -> usize {
        let mut res = 0;
        for i in 0..size {
            let value = perm[i];
            res += d[value][i];
        }
        res
    };
    let mut perm: Vec<_> = ids.iter().map(|&x| conv_id(start_permutation[x])).collect();
    if perm_parity(&perm) == 1 {
        return None;
    }
    let triangles: Vec<_> = triangles
        .iter()
        .map(|tr| {
            let cycle = tr.cycle();
            let mut new_cycle = vec![0; 3];
            for i in 0..3 {
                new_cycle[i] = conv_id(cycle[i]);
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

    // let mut all_cycles: Vec<_> = triangles.iter().map(|tr| tr.cycle()).collect();
    // all_cycles.sort();
    // for cycle in all_cycles.iter() {
    //     eprintln!("{:?}", cycle);
    // }

    // eprintln!("Perm: {:?}", perm);
    // let mut cur_score = score(&perm);
    // eprintln!("start score={}", cur_score);

    let mut path = a_star(&perm, &triangles, score);

    // loop {
    //     let mut changed = false;
    //     for need_delta in (1..=3).rev() {
    //         for tr in triangles.iter() {
    //             tr.mv.permutation.apply(&mut perm);
    //             let new_score = score(&perm);
    //             if new_score + need_delta <= cur_score {
    //                 cur_score = new_score;
    //                 changed = true;
    //                 eprintln!("new score={}. Need delta: {need_delta}", cur_score);
    //             } else {
    //                 tr.mv.permutation.apply_rev(&mut perm);
    //             }
    //         }
    //         if changed {
    //             break;
    //         }
    //     }
    //     if !changed {
    //         break;
    //     }
    // }
    Some(path)
}

#[derive(Clone)]
struct Prev {
    perm: Vec<usize>,
    edge_id: usize,
}

fn a_star(
    start: &[usize],
    triangles: &[Triangle],
    score: impl Fn(&[usize]) -> usize,
) -> Vec<usize> {
    let mut queue = BinaryHeap::new();
    queue.push(State::new(start.to_vec(), 0, score(start)));
    let mut seen = HashMap::new();
    seen.insert(
        start.to_vec(),
        Prev {
            perm: start.to_vec(),
            edge_id: usize::MAX,
        },
    );
    while let Some(state) = queue.pop() {
        // if it % 1000 == 0 || state.score < 3 {
        //     eprintln!(
        //         "It = {}, score = {}. Len = {}. Perm: {:?}",
        //         it,
        //         state.score,
        //         queue.len(),
        //         state.perm
        //     );
        // }
        if state.score == 0 {
            eprintln!("Found solution: {:?}", state.spent_moves);
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
        for (tr_id, tr) in triangles.iter().enumerate() {
            let mut new_perm = state.perm.clone();
            tr.mv.permutation.apply(&mut new_perm);
            let new_score = score(&new_perm);
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
