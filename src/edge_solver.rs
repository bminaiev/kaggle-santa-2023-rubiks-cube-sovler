use core::prelude::v1;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::format,
};

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    cube_edges_calculator::{
        build_squares, calc_cube_centers, calc_cube_edges, calc_edges_score, viz_edges_score,
    },
    edge_solver_dwalton::solve_edges_dwalton,
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    sol_utils::TaskSolution,
    utils::{calc_cube_side_size, get_cube_side_moves, show_cube_ids},
};

fn get_columns(sz: usize, delta: usize) -> [usize; 2] {
    if sz % 2 == 1 {
        [sz / 2 - delta - 1, sz / 2 + 1 + delta]
    } else {
        [sz / 2 - delta - 1, sz / 2 + delta]
    }
}

fn get_possible_moves(sol: &TaskSolution) -> Vec<Vec<SeveralMoves>> {
    let n = sol.task.info.n;
    let sz = calc_cube_side_size(n);
    let squares = build_squares(sz);
    let edges = calc_cube_edges(&squares);

    let puzzle_info = &sol.task.info;
    let cube_centers = calc_cube_centers(&squares);
    let side_moves = get_cube_side_moves(sz);

    (0..edges.len())
        .into_par_iter()
        .map(|delta| {
            let nums = get_columns(sz, delta);
            let mut res = vec![];
            for &num in nums.iter() {
                for &side_1 in ["r", "f", "d"].iter() {
                    let mv1 = format!("{side_1}{num}");
                    for side_mv1 in side_moves.iter() {
                        for side_mv2 in side_moves.iter() {
                            let moves = [
                                mv1.clone(),
                                side_mv1.to_owned(),
                                side_mv2.to_owned(),
                                rev_move(side_mv1),
                                rev_move(&mv1),
                                rev_move(side_mv2),
                            ];
                            let mut perm = Permutation::identity();

                            for mv in moves.iter() {
                                perm = perm.combine(&puzzle_info.moves[&mv.to_string()]);
                            }

                            let mut changes_center = false;
                            for cycle in perm.cycles.iter() {
                                for &v in cycle.iter() {
                                    if cube_centers[v] {
                                        changes_center = true;
                                    }
                                }
                            }

                            if changes_center {
                                continue;
                            }

                            res.push(SeveralMoves {
                                name: moves.to_vec(),
                                permutation: perm,
                            });
                        }
                    }
                }
            }
            res
        })
        .collect()
}

fn possible_to_make_centers_right(sol: &TaskSolution, state: &[usize]) -> Option<Vec<String>> {
    if !sol.exact_perm {
        return Some(vec![]);
    }
    let n = sol.task.info.n;
    let sz = calc_cube_side_size(n);
    let squares = build_squares(sz);
    let cube_centers = calc_cube_centers(&squares);
    let puzzle_info = &sol.task.info;

    let mut res = vec![];
    let mut state = state.to_vec();
    let side_moves = get_cube_side_moves(sz);

    let cnt_ok_centers = |a: &[usize]| {
        (0..cube_centers.len())
            .filter(|&i| cube_centers[i] && a[i] == sol.target_state[i])
            .count()
    };

    for side_mv in side_moves.iter() {
        for cnt in (1..=2).rev() {
            let cur_ok_centers = cnt_ok_centers(&state);
            let mut new_state = state.to_vec();
            for _ in 0..cnt {
                puzzle_info.moves[side_mv].apply(&mut new_state);
            }
            let new_ok_centers = cnt_ok_centers(&new_state);
            if new_ok_centers > cur_ok_centers {
                for _ in 0..cnt {
                    eprintln!(
                        "Apply {side_mv}. centers improved: {cur_ok_centers} -> {new_ok_centers}"
                    );
                    res.push(side_mv.to_string());
                }
                state = new_state;
            }
        }
    }
    let need_centers = (sz - 2) * (sz - 2) * 6;
    if cnt_ok_centers(&state) == need_centers {
        Some(res)
    } else {
        None
    }
}

fn try_solve_edges(
    sol: &TaskSolution,
    rng: &mut StdRng,
    possible_moves: &[Vec<SeveralMoves>],
    allow_retries: bool,
) -> Option<Vec<String>> {
    let n = sol.task.info.n;
    let sz = calc_cube_side_size(n);

    if sz == 3 {
        return Some(vec![]);
    }

    let squares = build_squares(sz);
    let edges = calc_cube_edges(&squares);

    let side_moves = get_cube_side_moves(sz);
    let mut answer = vec![];
    let puzzle_info = &sol.task.info;
    let mut state = sol.state.clone();
    if allow_retries {
        for _ in 0..rng.gen_range(0..5) {
            let mv = side_moves.choose(rng).unwrap().clone();
            puzzle_info.moves[&mv].apply(&mut state);
            answer.push(mv);
        }
    }
    // assert!(possible_to_make_centers_right(sol, &state).is_some());
    let puzzle_info = &sol.task.info;

    for (lvl, possible_moves) in possible_moves.iter().enumerate() {
        eprintln!("Lvl: {lvl}. Cnt moves: {}", possible_moves.len());
        let mut parity_changes = 0;
        loop {
            let mut edges_score = calc_edges_score(&edges, &state, &sol.target_state);
            loop {
                // eprintln!("Edges score: {:?}", edges_score);
                let mut changed = false;

                for mv in possible_moves.iter() {
                    let mut new_state = state.clone();
                    mv.permutation.apply(&mut new_state);
                    let new_edges_score = calc_edges_score(&edges, &new_state, &sol.target_state);
                    if new_edges_score > edges_score {
                        edges_score = new_edges_score;
                        eprintln!("New edges score: {edges_score:?}");
                        for mv in mv.name.iter() {
                            answer.push(mv.to_string());
                            puzzle_info.moves[mv].apply(&mut state);
                        }
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }
            }
            if edges_score[lvl] == edges[lvl].len() {
                eprintln!("FOUND SOLUTION FOR LVL: {lvl}");
                break;
            }
            // if !allow_retries {
            //     return None;
            // }
            let mut seen = HashMap::new();
            let mut cur_state = state.clone();
            seen.insert(cur_state.clone(), "".to_string());
            let mut queue = VecDeque::new();
            queue.push_back(cur_state);
            let mut iter = 100;
            let mut found = false;
            while let Some(cur_state) = queue.pop_front() {
                iter -= 1;
                if iter == 0 {
                    break;
                }
                for mv in side_moves.iter() {
                    let mut new_state = cur_state.clone();
                    puzzle_info.moves[mv].apply(&mut new_state);
                    // let new_edges_score = calc_edges_score(&edges, &new_state, &sol.target_state);
                    // if new_edges_score != edges_score {
                    //     eprintln!("Applied move: {mv}");
                    //     eprintln!("Prev edges score: {edges_score:?}");
                    //     show_cube_ids(&viz_edges_score(&edges, &cur_state, &sol.target_state), sz);
                    //     eprintln!("New edges score: {new_edges_score:?}");
                    //     show_cube_ids(&viz_edges_score(&edges, &new_state, &sol.target_state), sz);
                    //     unreachable!();
                    // }
                    if seen.contains_key(&new_state) {
                        continue;
                    }
                    seen.insert(new_state.clone(), mv.to_string());
                    queue.push_back(new_state.clone());

                    for mv2 in possible_moves.iter() {
                        let mut new_state = new_state.clone();
                        mv2.permutation.apply(&mut new_state);
                        let new_edges_score =
                            calc_edges_score(&edges, &new_state, &sol.target_state);
                        if new_edges_score > edges_score {
                            found = true;

                            // eprintln!(
                            //     "RESTART!!!!!!!! Potential score: {:?} -> {:?}",
                            //     edges_score, new_edges_score
                            // );
                            break;
                        }
                    }
                    if found {
                        let mut path = vec![];
                        loop {
                            let mv = seen[&new_state].clone();
                            if mv.is_empty() {
                                break;
                            }
                            puzzle_info.moves[&mv].apply_rev(&mut new_state);
                            path.push(mv);
                        }
                        path.reverse();
                        for mv in path.iter() {
                            answer.push(mv.to_string());
                            puzzle_info.moves[mv].apply(&mut state);
                        }
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                if sol.exact_perm || parity_changes >= 1 || !allow_retries {
                    eprintln!("Failed to find sol for lvl {lvl}");
                    return None;
                }
                parity_changes += 1;
                eprintln!("FAILED TO FIND SOLUTION FOR LVL: {lvl}... Let's try to change parity..");
                let nums = get_columns(sz, lvl);
                // TODO: try both?
                let basic_block = vec![
                    "d0".to_string(),
                    "d0".to_string(),
                    format!("r{}", nums.choose(rng).unwrap()),
                ];
                let moves = vec![basic_block; 5]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                for mv in moves.iter() {
                    answer.push(mv.to_string());
                    puzzle_info.moves[mv].apply(&mut state);
                }
            }
        }
    }

    Some(answer)
}

pub fn solve_edges(sol: &mut TaskSolution) -> bool {
    if !sol.exact_perm {
        return solve_edges_dwalton(sol);
    }
    let possible_moves = get_possible_moves(sol);
    // if sol.exact_perm {
    //     assert!(possible_to_make_centers_right(sol, &sol.state).is_some());
    // }

    let mut rng = StdRng::seed_from_u64(34534543);

    let allow_retries = false;
    for glob_iter in 0..if allow_retries { 100 } else { 1 } {
        eprintln!("START ITER: {glob_iter}");

        if let Some(moves) = try_solve_edges(sol, &mut rng, &possible_moves, allow_retries) {
            for mv in moves.iter() {
                sol.append_move(mv);
            }
            // TODO: only do this if needed
            // let extra_moves = possible_to_make_centers_right(sol, &sol.state).unwrap();
            // for mv in extra_moves.iter() {
            //     sol.append_move(mv);
            // }
            eprintln!("EDGES SOLVED!");
            sol.show();
            return true;
        }
    }
    eprintln!("EDGES NOT SOLVED :(");
    false
}
