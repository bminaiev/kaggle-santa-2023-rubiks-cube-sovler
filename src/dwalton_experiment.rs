use std::{
    collections::{BTreeSet, HashSet},
    process::Command,
};

use crate::{
    cube_edges_calculator::build_squares,
    data::Data,
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
    solver_nnn::solve_subproblem00,
    to_cube3_converter::Cube3Converter,
    utils::{
        calc_cube_side_size, conv_cube_to_dwalton, conv_dwalton_moves, show_cube_ids, DwaltonMove,
    },
};

#[derive(Clone, Debug)]
pub struct DwaltonLine {
    pub moves: Vec<DwaltonMove>,
    pub comment: String,
    pub changed_indexes: BTreeSet<usize>,
}

pub fn run_dwalton_solver_state(state: String) -> Option<Vec<DwaltonLine>> {
    let sz = calc_cube_side_size(state.len());

    eprintln!("State: {state}");
    let output = Command::new("./rubiks-cube-solver.py")
        .args(["--state", &state])
        .current_dir("/home/borys/santa-2023/dwalton76/rubiks-cube-NxNxN-solver")
        .output()
        .expect("failed to execute process");

    let output = String::from_utf8(output.stdout).unwrap();
    if !output.lines().any(|line| line.starts_with("Solution: ")) {
        eprintln!("No solution found..");
        return None;
    }

    let lines = std::fs::read_to_string("/tmp/rubiks-cube-NxNxN-solver/solution.txt").unwrap();
    let mut res = Vec::new();
    let mut cur_line: Vec<DwaltonMove> = vec![];
    for token in lines.split_ascii_whitespace() {
        match token.strip_prefix("COMMENT_") {
            Some(suffix) => {
                let mut changed_indexes = BTreeSet::new();
                changed_indexes.insert(0);
                changed_indexes.insert(sz - 1);
                for mv in cur_line.iter() {
                    let idx = mv.get_index();
                    changed_indexes.insert(idx);
                    changed_indexes.insert(sz - 1 - idx);
                }
                res.push(DwaltonLine {
                    changed_indexes,
                    moves: cur_line.clone(),
                    comment: suffix.to_string(),
                });
                cur_line.clear();
            }
            None => cur_line.extend(conv_dwalton_moves(sz, token)),
        }
    }
    Some(res)
}

pub fn run_dwalton_solver(sol: &TaskSolution) -> Option<Vec<DwaltonLine>> {
    let state = conv_cube_to_dwalton(sol);
    run_dwalton_solver_state(state)
}

pub fn solve_dwalton(
    data: &Data,
    task_type: &str,
    cube3_converter: &Cube3Converter,
    exact_perm: bool,
    log: &mut SolutionsLog,
) {
    println!("Solving nnn: {task_type}");

    let mut solutions = TaskSolution::all_by_type(data, task_type, exact_perm);
    let mut solutions: Vec<_> = solutions
        .into_iter()
        .filter(|t| t.task.get_color_type() == "A")
        .collect();
    // solutions.reverse();
    eprintln!("Tasks cnt: {}", solutions.len());
    solutions.truncate(1);
    let task_id = solutions[0].task_id;
    eprintln!("Solving id={task_id}");

    let puzzle_info = data.puzzle_info.get(task_type).unwrap();

    let n = puzzle_info.n;
    eprintln!("n={n}");
    let sz = calc_cube_side_size(n);

    let squares = build_squares(sz);

    if sz % 2 == 1 {
        solve_subproblem00(&mut solutions, puzzle_info, &squares);
    }

    let show_ids = |a: &[usize]| {
        show_cube_ids(a, sz);
    };

    for sol in solutions.iter_mut() {
        if sol.task_id == 283 {
            eprintln!("Skipping 283. Exact_perm: {}", sol.exact_perm);
            continue;
        }
        eprintln!("DWALTON: {}", conv_cube_to_dwalton(sol));

        let mut prev_lines = usize::MAX;
        loop {
            sol.print(data);
            show_ids(&sol.get_correct_colors_positions());

            let lines = run_dwalton_solver(sol).unwrap();
            if lines.is_empty() {
                break;
            }
            if lines.len() >= prev_lines {
                eprintln!("WTF? Strategy failed! First comment: {}", lines[0].comment);
                unreachable!();
            }
            prev_lines = lines.len();
            {
                let line = &lines[0];
                eprintln!("Comment: {}", line.comment);
                eprintln!("Moves: {:?}", line.moves);
                let mut important_indexes = HashSet::new();
                important_indexes.insert(0);
                important_indexes.insert(sz - 1);
                for mv in line.moves.iter() {
                    let idx = mv.get_index();
                    important_indexes.insert(idx);
                    important_indexes.insert(sz - 1 - idx);
                }
                if line.comment.contains("vertical_bars") {
                    for i in 0..sz {
                        important_indexes.insert(i);
                    }
                }
                for mv in line.moves.iter() {
                    match mv {
                        DwaltonMove::Simple(mv) => {
                            sol.append_move(mv);
                        }
                        DwaltonMove::Wide(prefix, r) => {
                            // if sz % 2 == 0 {
                            for x in r.clone() {
                                if important_indexes.contains(&x) {
                                    sol.append_move(&format!("{prefix}{}", x));
                                }
                            }
                            // } else {
                            //     sol.append_move(&format!("{prefix}{}", r.start()));
                            //     sol.append_move(&format!("{prefix}{}", r.end()));
                            // }
                        }
                    }
                }
            }
        }

        cube3_converter.solve(data, sol, false);
        sol.print(data);
        show_ids(&sol.get_correct_colors_positions());

        // log.append(sol);
    }
}
