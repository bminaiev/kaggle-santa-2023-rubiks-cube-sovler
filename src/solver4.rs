use std::hash::Hasher;

use crate::{
    data::Data,
    groups::{apply_precomputed_moves, get_groups, precompute_moves},
    moves::{create_moves, SeveralMoves},
    puzzle_type::PuzzleType,
    sol_utils::TaskSolution,
    utils::get_blocks,
};

fn create_move_groups(puzzle_info: &PuzzleType) -> Vec<Vec<SeveralMoves>> {
    let mut move_groups = vec![];
    move_groups.push(create_moves(
        puzzle_info,
        &[
            "d0", "d1", "d2", "d3", "f0", "f1", "f2", "f3", "r0", "r1", "r2", "r3",
        ],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &[
            "d0", "d1", "d2", "d3", "f0", "f1x2", "f2x2", "f3", "r0", "r1x2", "r2x2", "r3",
        ],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &[
            "d0", "d1x2", "d2x2", "d3", "f0", "f1x2", "f2x2", "f3", "r0", "r1x2", "r2x2", "r3",
        ],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &["d0", "d3", "f0", "f3", "r0", "r3"],
    ));
    move_groups.push(create_moves(puzzle_info, &[]));
    move_groups
}

pub fn solve4(data: &Data, task_type: &str) {
    let mut solutions = TaskSolution::all_by_type(data, task_type);
    eprintln!("Tasks cnt: {}", solutions.len());
    solutions.truncate(1);
    let task_id = solutions[0].task_id;
    eprintln!("Solving id={task_id}");

    let puzzle_info = data.puzzle_info.get(task_type).unwrap();

    let move_groups = create_move_groups(puzzle_info);
    eprintln!("Total move groups: {}", move_groups.len());

    let moves = puzzle_info.get_all_moves();
    let blocks = get_blocks(puzzle_info.n, &moves);

    for step in 0..3 {
        eprintln!("Step: {step}");
        let groups = get_groups(&blocks, &move_groups[step + 1]);
        eprintln!("Groups are calculated.");
        for (i, group) in groups.groups.iter().enumerate() {
            eprintln!("Group {i}: {:?}", group);
        }
        let mut calc_hash = |a: &[usize], debug: bool| groups.hash(a).finish();
        let prec = precompute_moves(puzzle_info.n, &move_groups[step], &mut calc_hash);
        for sol in solutions.iter_mut() {
            if sol.failed_on_stage.is_some() {
                continue;
            }
            if !apply_precomputed_moves(&mut sol.state, &prec, calc_hash, &mut sol.answer) {
                sol.failed_on_stage = Some(0);
                eprintln!("FAILED!");
            }
        }
    }
    for sol in solutions.iter() {
        sol.print(data);
    }
}
