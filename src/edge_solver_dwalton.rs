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
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    sol_utils::TaskSolution,
    utils::{
        calc_cube_side_size, conv_cube_to_dwalton, conv_dwalton_moves, get_cube_side_moves,
        show_cube_ids, DwaltonMove,
    },
};

fn run_dwalton_solver(sol: &TaskSolution) -> Vec<DwaltonMove> {
    let state = conv_cube_to_dwalton(sol);
    eprintln!("State: {state}");
    let output = Command::new("./rubiks-cube-solver.py")
        .args(["--state", &state])
        .current_dir("/home/borys/santa-2023/dwalton76/rubiks-cube-NxNxN-solver")
        .output()
        .expect("failed to execute process");
    let output = String::from_utf8(output.stdout).unwrap();
    for line in output.lines() {
        if let Some(solution) = line.strip_prefix("Solution: ") {
            let n = sol.task.info.n;
            let sz = calc_cube_side_size(n);

            eprintln!("Found sol: {solution}");

            return conv_dwalton_moves(sz, solution);
        }
    }
    unreachable!();
}

pub fn solve_edges_dwalton(sol: &mut TaskSolution) {
    assert!(!sol.exact_perm);
    let n = sol.task.info.n;
    let sz = calc_cube_side_size(n);
    if sz == 3 {
        return;
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
        let moves = run_dwalton_solver(sol);
        let mut found = false;
        for mv in moves.iter() {
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
            let new_edge_scores = calc_edges_score(&edges, &sol.state, &sol.target_state);
            let ok = (0..=lvl).all(|lvl| new_edge_scores[lvl] == edges[lvl].len());
            if ok {
                found = true;
                break;
            }
        }
        assert!(found);
    }
    let final_len = sol.answer.len();
    eprintln!("Solved edges in {} moves", final_len - start_len);
}
