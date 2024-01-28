use std::collections::{BTreeSet, HashMap};

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    moves::rev_move,
    permutation::Permutation,
    sol_utils::TaskSolution,
    triangle_solver::{Solver, Triangle, TriangleGroupSolver},
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct EstimateChange {
    delta: i32,
    key: String,
    mask1: u64,
    mask2: u64,
    side_mv: String,
}

fn is_good(use_moves1: &[String], use_moves2: &[String]) -> bool {
    for mv1 in use_moves1.iter() {
        for mv2 in use_moves2.iter() {
            if mv1 == mv2 || rev_move(mv1) == *mv2 {
                return false;
            }
        }
    }
    true
}

#[derive(Clone)]
struct SameKeyTriangles {
    moves1: Vec<String>,
    moves2: Vec<String>,
    solver_ids: Vec<Vec<usize>>,
    deltas: Vec<Vec<i32>>,
    key: String,
    side_mv: String,
    bad: Vec<Vec<bool>>,
}

impl SameKeyTriangles {
    fn choose_best_move(&self) -> EstimateChange {
        let mut best = (0, 0, 0);
        for mask1 in 1u64..(1 << self.moves1.len()) {
            let mut sum = 0;
            let mut mask2 = 0;
            let mut seen_solvers = BTreeSet::new();
            for j in 0..self.moves2.len() {
                let mut cur = 0;
                let mut ok = true;
                for i in 0..self.moves1.len() {
                    if mask1 & (1 << i) != 0 {
                        if self.bad[i][j] {
                            ok = false;
                            break;
                        }
                        let delta = self.deltas[i][j];
                        if seen_solvers.contains(&self.solver_ids[i][j]) {
                            ok = false;
                            break;
                        }
                        cur += delta;
                    }
                }
                if cur < 0 && ok {
                    mask2 |= 1 << j;
                    sum += cur;
                    for i in 0..self.moves1.len() {
                        if mask1 & (1 << i) != 0 {
                            seen_solvers.insert(self.solver_ids[i][j]);
                        }
                    }
                }
            }
            let cur = (sum, mask1, mask2);
            if cur < best {
                best = cur;
            }
        }

        EstimateChange {
            delta: best.0,
            mask1: best.1,
            mask2: best.2,
            key: self.key.clone(),
            side_mv: self.side_mv.clone(),
        }
    }

    fn build_moves(&self, change: &EstimateChange) -> Vec<String> {
        let mut moves = vec![];
        let mut use_moves1 = vec![];
        let mut use_moves2 = vec![];
        for i in 0..self.moves1.len() {
            if change.mask1 & (1 << i) != 0 {
                use_moves1.push(self.moves1[i].clone());
            }
        }
        for i in 0..self.moves2.len() {
            if change.mask2 & (1 << i) != 0 {
                use_moves2.push(self.moves2[i].clone());
            }
        }
        let all_moves = Triangle::gen_combination_moves(&use_moves1, &use_moves2, &change.side_mv);
        for mv in all_moves.into_iter() {
            moves.push(mv);
        }
        moves
    }

    fn changed_solvers(&self, change: &EstimateChange) -> Vec<usize> {
        let mut res = vec![];
        for i in 0..self.moves1.len() {
            if change.mask1 & (1 << i) != 0 {
                for j in 0..self.moves2.len() {
                    if change.mask2 & (1 << j) != 0 {
                        res.push(self.solver_ids[i][j]);
                    }
                }
            }
        }
        res.sort();
        res.dedup();
        res
    }

    fn update_triangle_delta(&mut self, tr: &Triangle, delta: i32) {
        for i in 0..self.moves1.len() {
            for j in 0..self.moves2.len() {
                if self.moves1[i] == tr.mv1 && self.moves2[j] == tr.mv2 {
                    self.deltas[i][j] = delta;
                    return;
                }
            }
        }
        panic!("Triangle not found!");
    }
}

pub fn solve_all_triangles(groups: &[Vec<Triangle>], sol: &mut TaskSolution, exact_perm: bool) {
    // let puzzle_info = &sol.task.info;

    let mut triangles_total_applied = 0;
    let mut triangles_groups_joined = 0;

    let mut triangles_by_key = std::collections::HashMap::<String, Vec<Triangle>>::new();
    for triangles in groups.iter() {
        for tr in triangles.iter() {
            triangles_by_key
                .entry(tr.key())
                .or_default()
                .push(tr.clone());
        }
    }
    let mut solver_id_by_triangle = HashMap::new();
    for i in 0..groups.len() {
        for tr in groups[i].iter() {
            solver_id_by_triangle.insert(tr.clone(), i);
        }
    }

    eprintln!("Create solvers for triangles.");
    let mut solvers: Vec<_> = groups
        .iter()
        .map(|triangles| TriangleGroupSolver::new(triangles, &sol.state, &sol.target_state))
        .collect();

    let mut triangles_by_key: HashMap<String, SameKeyTriangles> = triangles_by_key
        .into_par_iter()
        .map(|(key, triangles)| {
            let side_mv = triangles[0].side_mv.clone();
            let mut moves1 = Vec::new();
            let mut moves2 = Vec::new();
            for tr in triangles.iter() {
                moves1.push(tr.mv1.clone());
                moves2.push(tr.mv2.clone());
            }
            moves1.sort();
            moves1.dedup();
            moves2.sort();
            moves2.dedup();
            let mut solver_ids = vec![vec![usize::MAX; moves2.len()]; moves1.len()];
            let mut deltas = vec![vec![i32::MAX; moves2.len()]; moves1.len()];
            let mut bad = vec![vec![false; moves2.len()]; moves1.len()];
            for (i, mv1) in moves1.iter().enumerate() {
                for (j, mv2) in moves2.iter().enumerate() {
                    if mv1 == mv2 || mv1 == &rev_move(mv2) {
                        bad[i][j] = true;
                    }
                    for tr in triangles.iter() {
                        if tr.mv1 == *mv1 && tr.mv2 == *mv2 {
                            let solver_id = solver_id_by_triangle[tr];
                            let solver = &solvers[solver_id];
                            let cur_ans_len = solver.cur_answer_len as i32;
                            let mut nstate = sol.state.clone();
                            tr.mv.permutation.apply(&mut nstate);
                            let new_ans_len = solver.solve(&nstate, Solver::default()).len() as i32;
                            let delta = new_ans_len - cur_ans_len;
                            solver_ids[i][j] = solver_id;
                            deltas[i][j] = delta;
                        }
                    }
                    if deltas[i][j] == i32::MAX {
                        bad[i][j] = true;
                    }
                }
            }
            let same_key = SameKeyTriangles {
                moves1,
                moves2,
                solver_ids,
                deltas,
                key: key.clone(),
                side_mv,
                bad,
            };
            (key, same_key)
        })
        .collect();

    eprintln!("Total keys for triangles: {}", triangles_by_key.len());
    let lens: Vec<usize> = solvers.iter().map(|s| s.cur_answer_len).collect();
    eprintln!("Lens: {:?}", lens);
    let cur_sum_answer_lens = solvers.iter().map(|s| s.cur_answer_len).sum::<usize>();
    eprintln!("Cur sum answer lens: {}", cur_sum_answer_lens);

    loop {
        let mut estimate_changes: Vec<_> = triangles_by_key
            .par_iter()
            .map(|(key, same_key)| same_key.choose_best_move())
            .collect();
        estimate_changes.sort();
        let change = estimate_changes[0].clone();
        if change.delta >= 0 {
            break;
        }
        eprintln!(
            "Apply change: {change:?}. Total moves: {}x{}",
            change.mask1.count_ones(),
            change.mask2.count_ones()
        );
        let same_key = triangles_by_key.get(&change.key).unwrap();

        let all_moves = same_key.build_moves(&change);

        for mv in all_moves.into_iter() {
            sol.append_move(&mv);
        }
        let solver_ids = same_key.changed_solvers(&change);
        for &solver_id in solver_ids.iter() {
            let solver = &mut solvers[solver_id];
            solver.cur_answer_len = solver.solve(&sol.state, Solver::default()).len();
        }
        let sum_len = solvers.iter().map(|s| s.cur_answer_len).sum::<usize>();
        eprintln!("Sum len: {}", sum_len);
        let lens = solvers.iter().map(|s| s.cur_answer_len).collect::<Vec<_>>();
        eprintln!("Lens: {:?}", lens);

        let triangles_to_update: Vec<_> = solver_ids.iter().flat_map(|i| &groups[*i]).collect();
        let updates: Vec<_> = triangles_to_update
            .par_iter()
            .map(|tr| {
                let mut nstate = sol.state.clone();
                tr.mv.permutation.apply(&mut nstate);
                let solver = &solvers[solver_id_by_triangle[tr]];
                let new_ans_len = solver.solve(&nstate, Solver::default()).len();
                let delta = new_ans_len as i32 - solver.cur_answer_len as i32;
                (tr, delta)
            })
            .collect();
        for (tr, delta) in updates.into_iter() {
            let same_key = triangles_by_key.get_mut(&tr.key()).unwrap();
            same_key.update_triangle_delta(tr, delta);
        }
    }

    eprintln!("FALLBACK TO REGULAR SOLVER...");

    for (solver, group) in solvers.iter().zip(groups.iter()) {
        let moves = solver.solve(&sol.state, Solver::default());
        for tr_id in moves.into_iter() {
            for mv in group[tr_id].mv.name.iter() {
                sol.append_move(mv);
            }
        }
    }

    eprintln!(
        "Task_id = {}. Used moves for now: {}",
        sol.task_id,
        sol.answer.len()
    );
    eprintln!("Triangles total: {triangles_total_applied}. Groups: {triangles_groups_joined}");
}
