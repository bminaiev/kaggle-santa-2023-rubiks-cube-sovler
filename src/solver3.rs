use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    hash::Hasher,
};

use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    checker::check_solution,
    data::Data,
    groups::{apply_precomputed_moves, get_groups, precompute_moves, Edge},
    moves::{create_moves, SeveralMoves},
    puzzle_type::PuzzleType,
    sol_utils::TaskSolution,
    utils::{get_all_perms, get_blocks, get_start_permutation},
};

// Thistlethwaite's groups
fn create_move_groups(puzzle_info: &PuzzleType) -> Vec<Vec<SeveralMoves>> {
    let mut move_groups = vec![];
    move_groups.push(create_moves(
        puzzle_info,
        &["d0", "d1", "d2", "f0", "f1", "f2", "r0", "r1", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0", "d2", "f0", "f2", "r0", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0x2", "d2x2", "f0", "f2", "r0", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0x2", "d2x2", "f0x2", "f2x2", "r0", "r2"],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0x2", "d2x2", "f0x2", "f2x2", "r0x2", "r2x2"],
    ));
    move_groups.push(create_moves(puzzle_info, &[]));
    move_groups
}

pub fn solve(data: &Data, task_type: &str) {
    eprintln!("SOLVING {task_type}");
    let mut solutions = TaskSolution::all_by_type(data, task_type);
    let puzzle_info = data.puzzle_info.get(task_type).unwrap();

    let move_groups = create_move_groups(puzzle_info);
    eprintln!("Total move groups: {}", move_groups.len());
    let moves = puzzle_info.get_all_moves();
    let blocks = get_blocks(puzzle_info.n, &moves);

    for (i, block) in blocks.iter().enumerate() {
        eprintln!("Block {i}: {block:?}");
    }

    for (step, w) in move_groups.windows(2).enumerate() {
        eprintln!("Calculating groups... Step: {step}");
        let groups = get_groups(&blocks, &w[1]);
        eprintln!("Groups are calculated.");

        let hacks_info = step == 3;

        let mut calc_hash = |a: &[usize], debug: bool| {
            let mut hasher = groups.hash(a);
            if hacks_info {
                for block in blocks.iter() {
                    if block.len() != 3 {
                        continue;
                    }
                    let min = block.iter().map(|&x| a[x]).min().unwrap();
                    hasher.write_usize(min);
                }
            }
            hasher.finish()
        };
        let prec = precompute_moves(puzzle_info.n, &w[0], &mut calc_hash);
        eprintln!("Precumputed size: {}", prec.len());
        let mut cnt_ok = 0;
        for sol in solutions.iter_mut() {
            if sol.failed_on_stage.is_some() {
                continue;
            }
            if !apply_precomputed_moves(&mut sol.state, &prec, calc_hash, &mut sol.answer) {
                sol.failed_on_stage = Some(step);
            } else {
                cnt_ok += 1;
            }
        }
        eprintln!("Still ok solutions: {cnt_ok}/{}", solutions.len());
    }
    for sol in solutions.iter() {
        if sol.failed_on_stage.is_some() {
            continue;
        }
        let task = &data.puzzles[sol.task_id];
        eprintln!(
            "TASK: {}. SOL LEN={}, colors = {}, State={}",
            sol.task_id,
            sol.answer.len(),
            task.num_colors,
            task.convert_solution(&sol.answer)
        );
        check_solution(task, &sol.answer);
    }
}
