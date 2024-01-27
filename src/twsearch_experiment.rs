use std::collections::{BTreeMap, HashMap};

use crate::{
    cube_edges_calculator::build_squares,
    data::Data,
    moves::rev_move,
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
    solver_nnn::solve_subproblem00,
    to_cube3_converter::Cube3Converter,
    utils::{calc_cube_side_size, show_cube_ids},
};

pub fn solve_twsearch(
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
        // solve_subproblem00(&mut solutions, puzzle_info, &squares);
    }

    let show_ids = |a: &[usize]| {
        show_cube_ids(a, sz);
    };

    for sol in solutions.iter_mut() {
        if sol.task_id == 283 {
            eprintln!("Skipping 283. Exact_perm: {}", sol.exact_perm);
            continue;
        }

        let solution = data.solutions.my_280k[&sol.task_id].clone();

        let mut moves_conv = BTreeMap::new();
        for pos in 0..sz {
            for (my, first, last) in [("f", "F", "B"), ("r", "R", "L"), ("d", "D", "U")].iter() {
                if pos < sz / 2 {
                    let pos2 = if pos == 0 {
                        "".to_string()
                    } else {
                        (pos + 1).to_string()
                    };
                    moves_conv.insert(format!("{my}{pos}"), format!("{pos2}{first}"));
                } else {
                    let pos2 = if pos == sz - 1 {
                        "".to_string()
                    } else {
                        (sz - pos).to_string()
                    };
                    moves_conv.insert(format!("{my}{pos}"), format!("{pos2}{last}'"));
                }
            }
        }

        let mut extra_moves = HashMap::new();
        for (k, v) in moves_conv.iter() {
            eprintln!("{} -> {}", k, v);
            let rev_v = match v.strip_suffix('\'') {
                Some(v) => v.to_string(),
                None => format!("{v}'"),
            };
            extra_moves.insert(format!("-{k}"), rev_v);
        }
        moves_conv.extend(extra_moves);

        eprintln!("Sol: {solution:?}");

        let mut converted_solution = String::new();
        for mv in solution.iter().rev() {
            let mv = rev_move(mv);
            let mv = &moves_conv[&mv];
            converted_solution.push_str(mv);
            converted_solution.push(' ');
        }

        eprintln!("Converted sol: {converted_solution}");

        // cube3_converter.solve(data, sol, false);
        sol.print(data);
        show_ids(&sol.get_correct_colors_positions());

        // log.append(sol);
    }
}
