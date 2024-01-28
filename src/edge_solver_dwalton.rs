use core::prelude::v1;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::format,
    ops::Range,
    process::Command,
};

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    cube_edges_calculator::{
        build_squares, calc_cube_centers, calc_cube_edges, calc_edges_score, viz_edges_score,
    },
    dwalton_experiment::run_dwalton_solver,
    edge_solver::solve_edges,
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    sol_utils::TaskSolution,
    utils::{
        calc_cube_side_size, conv_cube_to_dwalton, conv_dwalton_moves, get_cube_side_moves,
        show_cube_ids, DwaltonMove,
    },
};

pub fn solve_edges_dwalton(sol: &mut TaskSolution) -> bool {
    assert!(!sol.exact_perm);
    let n = sol.task.info.n;
    let sz = calc_cube_side_size(n);
    if sz == 3 {
        return true;
    }
    let squares = build_squares(sz);
    let edges = calc_cube_edges(&squares);
    let start_len = sol.answer.len();
    for lvl in 0..edges.len() {
        let now_edge_scores = calc_edges_score(&edges, &sol.state, &sol.target_state);
        eprintln!("Edge scores: {now_edge_scores:?}");
        if now_edge_scores[lvl] == edges[lvl].len() {
            continue;
        }
        if let Some(lines) = run_dwalton_solver(sol) {
            let mut found = false;
            for line in lines.iter() {
                for mv in line.moves.iter() {
                    match mv {
                        DwaltonMove::Simple(mv) => {
                            sol.append_move(mv);
                        }
                        DwaltonMove::Wide(prefix, r) => {
                            if sz % 2 == 0 {
                                for x in r.clone() {
                                    sol.append_move(&format!("{prefix}{}", x));
                                }
                            } else {
                                sol.append_move(&format!("{prefix}{}", r.start()));
                                sol.append_move(&format!("{prefix}{}", r.end()));
                            }
                        }
                    }
                }
                let new_edge_scores = calc_edges_score(&edges, &sol.state, &sol.target_state);
                let ok = (0..=lvl).all(|lvl| new_edge_scores[lvl] == edges[lvl].len());
                if ok {
                    found = true;
                    break;
                }
            }
            if !found {
                eprintln!("WTF?");
                let new_edge_scores = calc_edges_score(&edges, &sol.state, &sol.target_state);

                show_cube_ids(&sol.get_correct_colors_positions(), sz);
                eprintln!("Edge scores: {new_edge_scores:?}");
                run_dwalton_solver(sol);
            }
            assert!(found);
        } else {
            return false;
        }
    }
    let final_len = sol.answer.len();
    eprintln!("Solved edges in {} moves", final_len - start_len);
    true
}
