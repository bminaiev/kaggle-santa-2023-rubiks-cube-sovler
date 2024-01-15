use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::format,
};

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::{
    cube_edges_calculator::{build_squares, calc_cube_centers, calc_cube_edges, calc_edges_score},
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    sol_utils::TaskSolution,
    utils::{calc_cube_side_size, get_cube_side_moves},
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

    let mut res = vec![vec![]; edges.len()];

    for delta in 0..res.len() {
        let nums = get_columns(sz, delta);
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

                        res[delta].push(SeveralMoves {
                            name: moves.to_vec(),
                            permutation: perm,
                        });
                    }
                }
            }
        }
    }
    res
}

pub fn solve_edges(sol: &mut TaskSolution) {
    let n = sol.task.info.n;
    let sz = calc_cube_side_size(n);

    let squares = build_squares(sz);
    let edges = calc_cube_edges(&squares);
    // for (lvl, edges) in edges.iter().enumerate() {
    //     eprintln!("LVL: {lvl}");
    //     for edge in edges.iter() {
    //         eprintln!("  {:?}", edge);
    //     }
    // }
    let puzzle_info = &sol.task.info;

    let side_moves = get_cube_side_moves(sz);

    let possible_moves = get_possible_moves(sol);

    let mut rng = StdRng::seed_from_u64(7787788);

    for (lvl, possible_moves) in possible_moves.iter().enumerate() {
        eprintln!("Lvl: {lvl}. Cnt moves: {}", possible_moves.len());
        loop {
            let mut edges_score = calc_edges_score(&edges, &sol.state);
            loop {
                // eprintln!("Edges score: {:?}", edges_score);
                let mut changed = false;

                for mv in possible_moves.iter() {
                    let mut new_state = sol.state.clone();
                    mv.permutation.apply(&mut new_state);
                    let new_edges_score = calc_edges_score(&edges, &new_state);
                    if new_edges_score > edges_score {
                        edges_score = new_edges_score;
                        eprintln!("New edges score: {edges_score:?}");
                        for mv in mv.name.iter() {
                            sol.answer.push(mv.to_string());
                            puzzle_info.moves[mv].apply(&mut sol.state);
                        }
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }
            }
            if edges_score[lvl] == edges[lvl].len() / 2 {
                eprintln!("FOUND SOLUTION FOR LVL: {lvl}");
                break;
            }
            let mut seen = HashMap::new();
            let mut cur_state = sol.state.clone();
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
                    assert_eq!(calc_edges_score(&edges, &new_state), edges_score);
                    if seen.contains_key(&new_state) {
                        continue;
                    }
                    seen.insert(new_state.clone(), mv.to_string());
                    queue.push_back(new_state.clone());

                    for mv2 in possible_moves.iter() {
                        let mut new_state = new_state.clone();
                        mv2.permutation.apply(&mut new_state);
                        let new_edges_score = calc_edges_score(&edges, &new_state);
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
                        // sol.state = new_state.clone();
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
                            sol.answer.push(mv.to_string());
                            puzzle_info.moves[mv].apply(&mut sol.state);
                        }
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                eprintln!("FAILED TO FIND SOLUTION FOR LVL: {lvl}... Let's try to change parity..");
                let nums = get_columns(sz, lvl);
                // TODO: try both?
                let basic_block = vec![
                    "d0".to_string(),
                    "d0".to_string(),
                    format!("r{}", nums.choose(&mut rng).unwrap()),
                ];
                let moves = vec![basic_block; 5]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                for mv in moves.iter() {
                    sol.answer.push(mv.to_string());
                    puzzle_info.moves[mv].apply(&mut sol.state);
                }

                // break;
            }
        }
    }

    // TODO: only do this if needed
    let cube_centers = calc_cube_centers(&squares);
    for side_mv in side_moves.iter() {
        for cnt in 1..=2 {
            let cur_ok_centers = (0..cube_centers.len())
                .filter(|&i| cube_centers[i] && sol.state[i] == i)
                .count();
            let mut new_state = sol.state.clone();
            for _ in 0..cnt {
                puzzle_info.moves[side_mv].apply(&mut new_state);
            }
            let new_ok_centers = (0..cube_centers.len())
                .filter(|&i| cube_centers[i] && new_state[i] == i)
                .count();
            if new_ok_centers > cur_ok_centers {
                for _ in 0..cnt {
                    eprintln!(
                        "Apply {side_mv}. centers improved: {cur_ok_centers} -> {new_ok_centers}"
                    );
                    sol.answer.push(side_mv.to_string());
                    puzzle_info.moves[side_mv].apply(&mut sol.state);
                }
            }
        }
    }
}
