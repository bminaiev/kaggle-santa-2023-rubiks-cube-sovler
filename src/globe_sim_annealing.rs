use std::{collections::HashSet, ops::RangeBounds};

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    data::Data,
    globe_bfs::Recolor,
    globe_jaapsch::{globe_final_rows_move, GlobeSaStage, GlobeSolutionInfo, GlobeState},
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
    utils::pick_random_perm,
};

fn solve_rows(state: &GlobeState, sol: &mut TaskSolution, row1: usize) {
    let mut rng = StdRng::seed_from_u64(787788);

    let row2 = state.n_rows - row1 - 1;
    let mut recolor = Recolor::default();
    for i in 0..2 {
        for c in 0..state.n_cols {
            let r = [row1, row2][i];
            let pos = state.rc_to_index(r, c);
            recolor.get_or_add(sol.target_state[pos]);
        }
    }
    let mut conv_state = |sol: &TaskSolution| -> Vec<Vec<usize>> {
        let mut a = vec![vec![usize::MAX; state.n_cols]; 2];
        for i in 0..2 {
            for c in 0..state.n_cols {
                let r = [row1, row2][i];
                let pos = state.rc_to_index(r, c);
                a[i][c] = recolor.get(sol.state[pos]) as usize;
            }
        }
        a
    };

    let a = conv_state(sol);

    let num_colors = recolor.num_colors();
    let calc_score = |a: &[Vec<usize>]| -> usize {
        let mut res = 0;
        for i in 0..2 {
            let in_lis = calc_cycle_lis(&a[i], &|x| {
                x >= i * (num_colors / 2) && x <= (i + 1) * (num_colors / 2)
            });
            res = res.max(in_lis.iter().filter(|&&x| !x).count());
        }
        res
    };

    let show_rows = |a: &[Vec<usize>]| {
        let line = num_colors / 2;
        eprintln!("Score: {}", calc_score(a));
        for i in 0..a.len() {
            let in_lis = calc_cycle_lis(&a[i], &|x| x >= i * line && x < (i + 1) * line);
            eprintln!(
                "{}",
                a[i].iter()
                    .enumerate()
                    .map(|(pos, &x)| {
                        let is_good = in_lis[pos];
                        format!(
                            "{}{:2}{}",
                            if is_good { "[" } else { " " },
                            x,
                            if is_good { "]" } else { " " }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    };

    show_rows(&a);

    let mut build_stage = |a: &[Vec<usize>], rng: &mut StdRng| -> GlobeSaStage {
        let mut in_row = vec![vec![]; 2];
        for row_id in 0..2 {
            let in_lis = calc_cycle_lis(&a[row_id], &|x| {
                x >= row_id * (num_colors / 2) && x < (row_id + 1) * (num_colors / 2)
            });
            let offset = row_id * state.n_cols;
            for i in 0..a[row_id].len() {
                if !in_lis[i] {
                    in_row[row_id].push(offset + i);
                }
            }
        }
        eprintln!("In row: {in_row:?}");
        for row_id in 0..2 {
            while in_row[row_id].len() < in_row[1 - row_id].len() {
                let value = rng.gen_range(row_id * state.n_cols..(row_id + 1) * state.n_cols);
                if !in_row[row_id].contains(&value) {
                    in_row[row_id].push(value);
                }
            }
        }
        let mut stage = GlobeSaStage { in_row };
        eprintln!("Stage: {stage:?}");
        stage
    };

    let stage = build_stage(&a, &mut rng);

    let stages = vec![stage];

    let perms: Vec<_> = stages
        .iter()
        .map(|stage| pick_random_perm(stage.in_row[0].len(), &mut rng))
        .collect();

    let mut init_colors = vec![usize::MAX; state.n_cols * 2];
    for i in 0..2 {
        for c in 0..state.n_cols {
            let pos = i * state.n_cols + c;
            init_colors[pos] = a[i][c];
        }
    }

    let mut sol_info = GlobeSolutionInfo::new(stages, init_colors, &mut rng);

    sol_info.run_sa(&mut rng, &calc_score);
    let matching_res = sol_info.eval(calc_score);
    state.apply_matching_res(sol, &matching_res, row1);

    let mut a = conv_state(sol);

    show_rows(&a);

    for stage_it in 2..5 {
        eprintln!(
            "Stage {stage_it}! Current solution len: {}",
            sol.answer.len()
        );
        let stage = build_stage(&a, &mut rng);
        eprintln!("New stage: {stage:?}");
        let stage_back = GlobeSaStage {
            in_row: vec![stage.in_row[1].clone(), stage.in_row[0].clone()],
        };
        let stages = vec![stage, stage_back];
        let mut init_colors = vec![usize::MAX; state.n_cols * 2];
        for i in 0..2 {
            for c in 0..state.n_cols {
                let pos = i * state.n_cols + c;
                init_colors[pos] = a[i][c];
            }
        }
        let mut sol_info = GlobeSolutionInfo::new(stages, init_colors, &mut rng);
        sol_info.run_sa(&mut rng, &calc_score);
        let matching_res = sol_info.eval(calc_score);
        state.apply_matching_res(sol, &matching_res, row1);

        a = conv_state(sol);

        show_rows(&a);
    }
    eprintln!("Last solution len: {}", sol.answer.len());
}

fn calc_cycle_lis(a: &[usize], filter: &impl Fn(usize) -> bool) -> Vec<bool> {
    let mut res = vec![false; a.len()];
    for offset in 0..a.len() {
        let mut b = a.to_vec();
        b.rotate_left(offset);
        let mut cur_res = calc_lis(&b, filter);
        if cur_res.iter().filter(|&&x| x).count() > res.iter().filter(|&&x| x).count() {
            cur_res.rotate_right(offset);
            res = cur_res;
        }
    }
    res
}

fn calc_lis(a: &[usize], filter: &impl Fn(usize) -> bool) -> Vec<bool> {
    let mut dp = vec![0; a.len()];
    let mut prev = vec![usize::MAX; a.len()];
    let mut lis = vec![];
    for i in 0..a.len() {
        if !filter(a[i]) {
            continue;
        }
        let pos = dp[..lis.len()]
            .binary_search_by(|&x| {
                if x <= a[i] {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|x| x);
        dp[pos] = a[i];
        if pos == lis.len() {
            lis.push(i);
        } else {
            lis[pos] = i;
        }
        if pos > 0 {
            prev[i] = lis[pos - 1];
        }
    }
    let mut res = vec![false; a.len()];
    if let Some(mut cur) = lis.last().copied() {
        while cur != usize::MAX {
            res[cur] = true;
            cur = prev[cur];
        }
    }
    res
}

pub fn solve_globe_sa(data: &Data, task_types: &[&str], log: &mut SolutionsLog) {
    let mut solutions = TaskSolution::all_by_types(data, task_types);
    eprintln!("Number of tasks: {}", solutions.len());
    solutions.truncate(1);

    for sol in solutions.iter_mut() {
        let mut rng = StdRng::seed_from_u64(785834334);
        eprintln!(
            "Solving task {}. Type: {}, {}",
            sol.task_id,
            sol.task.puzzle_type,
            sol.task.get_color_type()
        );

        let puzzle_type = &data.puzzle_info[&sol.task.puzzle_type];
        let mut state = GlobeState::new(puzzle_type);

        for row in 0..state.n_rows / 2 {
            solve_rows(&state, sol, row);
            unreachable!();
        }

        globe_final_rows_move(&state, sol);
        state.show_state(&sol.state, &sol.target_state);
        assert!(sol.is_solved());
        eprintln!("Sol len [task={}]: {}", sol.task_id, sol.answer.len());
    }
    for sol in solutions.iter() {
        if sol.is_solved_with_wildcards() {
            // log.append(sol);
        } else {
            eprintln!("Failed to solve task {}?!", sol.task_id);
        }
    }
}
