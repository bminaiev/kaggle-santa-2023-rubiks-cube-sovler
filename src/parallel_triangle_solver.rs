use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    sol_utils::TaskSolution,
    triangle_solver::{Solver, Triangle, TriangleGroupSolver},
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EstimateChange {
    change: usize,
    key: String,
}

pub fn solve_all_triangles(groups: &[Vec<Triangle>], sol: &mut TaskSolution, exact_perm: bool) {
    let puzzle_info = &sol.task.info;

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

    eprintln!("Create solvers for triangles.");
    let mut solvers: Vec<_> = groups
        .iter()
        .map(|triangles| TriangleGroupSolver::new(triangles, &sol.state, &sol.target_state))
        .collect();

    eprintln!("Total keys: {}", triangles_by_key.len());
    let lens: Vec<usize> = solvers.iter().map(|s| s.cur_answer_len).collect();
    eprintln!("Lens: {:?}", lens);
    let cur_sum_answer_lens = solvers.iter().map(|s| s.cur_answer_len).sum::<usize>();
    eprintln!("Cur sum answer lens: {}", cur_sum_answer_lens);

    loop {
        let mut changed = false;
        for min_triangles_apply in (1..=3).rev() {
            if changed {
                // break;
            }
            eprintln!("Start new iteration... Min tr: {min_triangles_apply}");
            let mut estimate_changes: Vec<_> = triangles_by_key
                .keys()
                .map(|key| {
                    let mut change = 0;
                    for (solver, triangles) in solvers.iter().zip(groups.iter()) {
                        let mut here_triangles = vec![];
                        for tr in triangles.iter() {
                            if tr.key() == *key {
                                here_triangles.push(tr);
                            }
                        }
                        if here_triangles.is_empty() {
                            continue;
                        }
                        let cur_dist_estimate = solver.get_dist_estimate(&sol.state);
                        let mut best_dist = 0;
                        if cur_dist_estimate == 0 {
                            continue;
                        }
                        for tr in here_triangles.iter() {
                            let mut new_state = sol.state.clone();
                            tr.mv.permutation.apply(&mut new_state);
                            let new_dist_estimate = solver.get_dist_estimate(&new_state);
                            if new_dist_estimate < cur_dist_estimate {
                                let change_here = cur_dist_estimate - new_dist_estimate;
                                if change_here > best_dist {
                                    best_dist = change_here;
                                }
                            }
                        }
                        change += best_dist;
                    }
                    EstimateChange {
                        change,
                        key: key.clone(),
                    }
                })
                .collect();
            estimate_changes.sort();
            estimate_changes.reverse();

            for change in estimate_changes.iter() {
                let mut apply_triangles: Vec<&Triangle> = vec![];
                let mut groups_changed = vec![];
                let changes_to_apply: Vec<_> = (0..solvers.len())
                    .into_par_iter()
                    .flat_map(|i| {
                        let solver = &solvers[i];
                        let triangles = &groups[i];
                        let mut here_triangles = vec![];
                        for tr in triangles.iter() {
                            if tr.key() == change.key {
                                here_triangles.push(tr);
                            }
                        }
                        if here_triangles.is_empty() {
                            return None;
                        }
                        let cur_dist_estimate = solver.get_dist_estimate(&sol.state);
                        if cur_dist_estimate == 0 {
                            return None;
                        }
                        let mut best_dist = (0, None);
                        for tr in here_triangles.iter() {
                            let mut new_state = sol.state.clone();
                            tr.mv.permutation.apply(&mut new_state);
                            let new_dist_estimate = solver.get_dist_estimate(&new_state);
                            if new_dist_estimate < cur_dist_estimate {
                                let real_ans_len =
                                    solver.solve(&new_state, Solver::default()).len();
                                if real_ans_len < solver.cur_answer_len {
                                    let dist = solver.cur_answer_len - real_ans_len;
                                    if dist > best_dist.0 {
                                        best_dist = (dist, Some(tr));
                                    }
                                }
                            }
                        }
                        if best_dist.0 > 0 {
                            let tr_to_use = best_dist.1.unwrap();
                            let can_use =
                                apply_triangles.iter().all(|tr| tr_to_use.can_combine(tr));
                            if can_use {
                                return Some((i, *tr_to_use));
                                // apply_triangles.push(tr_to_use);
                                // groups_changed.push(i);
                            }
                        }
                        None
                    })
                    .collect();
                for (i, tr) in changes_to_apply.into_iter() {
                    apply_triangles.push(tr);
                    groups_changed.push(i);
                }
                if apply_triangles.len() >= min_triangles_apply {
                    changed = true;
                    triangles_total_applied += apply_triangles.len();
                    triangles_groups_joined += 1;
                    // for tt in apply_triangles.iter() {
                    //     // eprintln!("Apply triangle: {}", tt.mv.name);
                    // }
                    let all_moves = Triangle::gen_combination_moves(&apply_triangles);
                    for mv in all_moves.into_iter() {
                        puzzle_info.moves[&mv].apply(&mut sol.state);
                        sol.answer.push(mv.clone());
                    }
                    for &gr in groups_changed.iter() {
                        let new_ans = solvers[gr].solve(&sol.state, Solver::default()).len();
                        assert!(new_ans < solvers[gr].cur_answer_len);
                        solvers[gr].cur_answer_len = new_ans;
                    }
                    let new_sum_len = solvers.iter().map(|s| s.cur_answer_len).sum::<usize>();
                    eprintln!(
                        "New sum len: {}. Ans len: {}. Av group size: {}",
                        new_sum_len,
                        sol.answer.len(),
                        triangles_total_applied as f64 / triangles_groups_joined as f64
                    );
                    // break;
                }
            }
        }
        if !changed {
            break;
        }
    }

    eprintln!("FALLBACK TO REGULAR SOLVER...");

    for (solver, group) in solvers.iter().zip(groups.iter()) {
        let moves = solver.solve(&sol.state, Solver::default());
        for tr_id in moves.into_iter() {
            for mv in group[tr_id].mv.name.iter() {
                puzzle_info.moves[&mv.to_string()].apply(&mut sol.state);
                sol.answer.push(mv.clone());
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
