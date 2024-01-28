use std::collections::HashSet;

use rand::{rngs::StdRng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    cube_edges_calculator::{build_squares, calc_cube_edges, calc_edges_score},
    data::Data,
    dsu::Dsu,
    dwalton_experiment::run_dwalton_solver_state,
    edge_solver::solve_edges,
    greedy::greedy_cube_optimizer,
    parallel_triangle_solver::{solve_all_triangles, solve_all_triangles_greedy},
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
    solver_nnn::solve_subproblem00,
    to_cube3_converter::Cube3Converter,
    triangle_solver::Triangle,
    triangles_parity::triangle_parity_solver,
    utils::{
        calc_cube_side_size, conv_colors_to_dwalton, get_cube_side_moves, show_cube_ids,
        DwaltonMove,
    },
};

pub fn solve_exact_perm(
    data: &Data,
    task_type: &str,
    cube3_converter: &Cube3Converter,
    log: &mut SolutionsLog,
) {
    println!("Solving nnn: {task_type}");
    let exact_perm = true;

    let mut solutions = TaskSolution::all_by_type(data, task_type, exact_perm);
    let mut solutions: Vec<_> = solutions
        .into_iter()
        .filter(|t| t.task.get_color_type() == "N1")
        .collect();
    // solutions.reverse();
    eprintln!("Tasks cnt: {}", solutions.len());
    // solutions.swap(0, 1);
    // solutions.truncate(1);
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

    let side_moves = get_cube_side_moves(sz);
    let keys = puzzle_info.moves.keys().collect::<Vec<_>>();

    let mut rng = StdRng::seed_from_u64(345345);

    let mv_side = |mv: &str| -> String {
        if mv.starts_with('-') {
            mv[1..2].to_string()
        } else {
            mv[0..1].to_string()
        }
    };

    let par_maps: Vec<_> = keys
        .par_iter()
        .map(|&mv1| {
            if side_moves.contains(&mv1.to_string()) {
                return vec![];
            }
            let mut moves = vec![];
            for mv2 in puzzle_info.moves.keys() {
                if side_moves.contains(&mv2.to_string()) {
                    continue;
                }
                for side_mv in side_moves.iter() {
                    if mv_side(mv1) == mv_side(side_mv) || mv_side(mv2) == mv_side(side_mv) {
                        continue;
                    }
                    if let Some(tr) = Triangle::create(puzzle_info, mv1, mv2, side_mv) {
                        moves.push(tr);
                    }
                }
            }
            moves
        })
        .collect();
    let mut dsu = Dsu::new(n);
    let par_maps: Vec<_> = par_maps.into_iter().flatten().collect();
    for tr in par_maps.iter() {
        for w in tr.mv.permutation.cycles[0].windows(2) {
            dsu.unite(w[0], w[1]);
        }
    }
    let mut triangles_by_dsu = vec![vec![]; n];
    for tr in par_maps.iter() {
        let id = dsu.get(tr.mv.permutation.cycles[0][0]);
        triangles_by_dsu[id].push(tr.clone());
    }

    let triangle_groups: Vec<_> = triangles_by_dsu
        .into_iter()
        .filter(|x| !x.is_empty())
        .collect();

    let show_ids = |a: &[usize]| {
        show_cube_ids(a, sz);
    };

    for sol in solutions.iter_mut() {
        if sol.task_id == 283 {
            eprintln!("Skipping 283. Exact_perm: {}", sol.exact_perm);
            continue;
        }
        // eprintln!("DWALTON: {}", conv_cube_to_dwalton(sol));

        // eprintln!("State: {:?}", sol.state);

        let edges = calc_cube_edges(&squares);

        let mut state = vec![0; 6 * sz * sz];
        for i in 0..state.len() {
            state[i] = i / sz / sz;
        }
        let solution = data.solutions.s853k[&sol.task_id].clone();
        for mv in solution.iter().rev() {
            let perm = &puzzle_info.moves[mv];
            perm.apply_rev(&mut state);
        }
        for mv in sol.answer.iter() {
            let perm = &puzzle_info.moves[mv];
            perm.apply(&mut state);
        }
        let dwalton = conv_colors_to_dwalton(&state);
        eprintln!("DWALTON: {}", dwalton);
        let res = run_dwalton_solver_state(dwalton).unwrap();
        eprintln!("Got dwalton solution");
        for line in res.iter() {
            eprintln!("Line: {:?}", line);
            for mv in line.moves.iter() {
                match mv {
                    DwaltonMove::Simple(mv) => {
                        sol.append_move(mv);
                    }
                    DwaltonMove::Wide(prefix, r) => {
                        for x in r.clone() {
                            sol.append_move(&format!("{prefix}{}", x));
                        }
                    }
                }
            }
        }
        eprintln!("DWALTON solution applied");
        show_ids(&sol.get_correct_colors_positions());

        // let edges_score = calc_edges_score(&edges, &state, &sol.target_state);
        // let ok_edges = edges_score
        //     .iter()
        //     .zip(edges.iter())
        //     .all(|(score, edges)| *score == edges.len());
        // eprintln!("Edges score: {:?}", edges_score);
        // assert!(ok_edges);

        eprintln!("Solving last triangles...");

        solve_all_triangles(&triangle_groups, sol, exact_perm);

        eprintln!("Before solving edges... They should be solved?");

        // show_ids(&sol.get_correct_colors_positions());

        solve_edges(sol);

        sol.print(data);
        show_ids(&sol.get_correct_colors_positions());

        // cube3_converter.solve(data, sol, exact_perm);
        // sol.print(data);
        // show_ids(&sol.get_correct_colors_positions());

        assert!(sol.is_solved_with_wildcards());
        log.append(sol);
    }
}
