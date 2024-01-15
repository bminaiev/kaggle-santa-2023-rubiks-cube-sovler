use std::{
    collections::{BTreeSet, HashSet},
    mem::needs_drop,
    ops::Sub,
};

use rand::seq::SliceRandom;
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    vec,
};

use crate::{
    cube_edges_calculator::{build_squares, calc_cube_centers, calc_cube_edges, calc_edges_score},
    data::Data,
    dsu::Dsu,
    edge_solver::solve_edges,
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    puzzle::Puzzle,
    puzzle_type::PuzzleType,
    sol_utils::TaskSolution,
    to_cube3_converter::Cube3Converter,
    triangle_solver::{solve_triangle, Triangle},
    triangles_parity::triangle_parity_solver,
    utils::{calc_cube_side_size, get_cube_side_moves},
};

struct Subspace {
    interesting_positions: Vec<usize>,
}

impl Subspace {
    fn new(mut interesting_positions: Vec<usize>) -> Self {
        interesting_positions.sort();
        interesting_positions.dedup();
        Self {
            interesting_positions,
        }
    }

    fn conv_pos(&self, x: usize) -> Option<usize> {
        self.interesting_positions.binary_search(&x).ok()
    }

    fn get_moves(&self, puzzle_info: &PuzzleType, forbidden_moves: &[String]) -> Vec<SeveralMoves> {
        let mut res = vec![];
        for (k, perm) in puzzle_info.moves.iter() {
            if forbidden_moves.contains(k) {
                continue;
            }
            let mut interesting_move = false;
            for &cell_id in self.interesting_positions.iter() {
                for cycle in perm.cycles.iter() {
                    if cycle.contains(&cell_id) {
                        interesting_move = true;
                        break;
                    }
                }
            }
            if interesting_move {
                res.push((k.clone(), perm.clone()));
            }
        }
        self.conv_moves(&res)
    }

    fn conv_moves(&self, moves: &[(String, Permutation)]) -> Vec<SeveralMoves> {
        let mut res = vec![];
        for (k, perm) in moves.iter() {
            let mut cycles = vec![];
            for cycle in perm.cycles.iter() {
                let mut new_cycle = vec![];
                for &id in cycle.iter() {
                    if let Some(index) = self.conv_pos(id) {
                        new_cycle.push(index);
                    }
                }
                if !new_cycle.is_empty() {
                    assert_eq!(new_cycle.len(), cycle.len());
                    cycles.push(new_cycle);
                }
            }
            res.push(SeveralMoves {
                name: vec![k.clone()],
                permutation: Permutation { cycles },
            });
        }
        res
    }

    fn conv_state(&self, state: &[usize]) -> Vec<usize> {
        self.interesting_positions
            .iter()
            .map(|&x| self.conv_pos(state[x]).unwrap())
            .collect::<Vec<_>>()
    }
}

fn dfs(
    more_layers: usize,
    state: &mut [usize],
    moves: &[SeveralMoves],
    estimate_dist: &impl Fn(&[usize]) -> usize,
    iter: &mut usize,
    moves_history: &mut Vec<usize>,
) -> bool {
    *iter += 1;
    if *iter % 10_000_000 == 0 {
        eprintln!("ITER: {}", *iter);
    }
    let dist = estimate_dist(state);
    if dist == 0 {
        return true;
    }
    if dist > more_layers {
        return false;
    }
    for (mov_id, mov) in moves.iter().enumerate() {
        if mov.name.len() > more_layers {
            continue;
        }
        mov.permutation.apply(state);
        moves_history.push(mov_id);
        if dfs(
            more_layers - mov.name.len(),
            state,
            moves,
            estimate_dist,
            iter,
            moves_history,
        ) {
            return true;
        }
        mov.permutation.apply_rev(state);
        moves_history.pop();
    }
    false
}

fn calc_positions(squares: &[Vec<Vec<usize>>], dx: usize, dy: usize) -> Vec<usize> {
    let center = squares[0].len() / 2;
    let mut res = BTreeSet::new();
    for square in squares.iter() {
        for &(r, c) in [
            (center - dx, center - dy),
            (center - dx, center + dy),
            (center + dx, center - dy),
            (center + dx, center + dy),
            (center - dy, center - dx),
            (center - dy, center + dx),
            (center + dy, center - dx),
            (center + dy, center + dx),
        ]
        .iter()
        {
            res.insert(square[r][c]);
        }
    }
    res.into_iter().collect::<Vec<_>>()
}

pub fn solve_subproblem(
    solutions: &mut [TaskSolution],
    puzzle_info: &PuzzleType,
    squares: &[Vec<Vec<usize>>],
    dx: usize,
    dy: usize,
) {
    if dx == 0 && dy == 0 {
        solve_subproblem00(solutions, puzzle_info, squares);
        return;
    }
    let mut interesting_positions = calc_positions(squares, dx, dy);
    // let mut more = calc_positions(squares, 0, 0);
    // interesting_positions.append(&mut more);
    let subspace = Subspace::new(interesting_positions);
    let sz = squares[0].len();
    let first_sq_positions: Vec<_> = (0..subspace.interesting_positions.len())
        .filter(|&pos| {
            let x = subspace.interesting_positions[pos];
            x > sz * sz && x < sz * sz * 2
        })
        .collect();
    let mut forbidden_moves = vec![];
    for mv in ["d", "f", "r"] {
        for sign in ["", "-"] {
            let name = format!("{sign}{mv}{}", sz / 2);
            forbidden_moves.push(name);
        }
    }
    eprintln!("FORBIDDEN MOVES: {forbidden_moves:?}");
    let moves = subspace.get_moves(puzzle_info, &forbidden_moves);
    eprintln!("Solve subproblem {dx}, {dy}. Moves: {}", moves.len());
    for mv in moves.iter() {
        eprintln!("Move: {:?}", mv.name);
    }
    for sol in solutions.iter_mut() {
        let state: Vec<_> = subspace
            .interesting_positions
            .iter()
            .map(|&x| sol.task.solution_state[sol.state[x]])
            .collect();
        eprintln!("Before conv: {state:?}");
        // let state = sol.task.convert_state_to_colors(&state);
        let target_state = sol
            .task
            .convert_state_to_colors(&subspace.interesting_positions);
        eprintln!("Target state: {target_state:?}");
        // eprintln!("my state: {state:?}");

        for max_layers in 0.. {
            eprintln!("max_layers={max_layers}");
            let dist = |a: &[usize]| -> usize {
                for &i in first_sq_positions.iter() {
                    if a[i] != target_state[i] {
                        return 1;
                    }
                }
                0
            };
            let mut moves_history = vec![];
            if dfs(
                max_layers,
                &mut state.clone(),
                &moves,
                &dist,
                &mut 0,
                &mut moves_history,
            ) {
                eprintln!("Found solution in {} moves!", moves_history.len());
                for &mov_id in moves_history.iter() {
                    for move_id in moves[mov_id].name.iter() {
                        let full_perm = &puzzle_info.moves[move_id];
                        full_perm.apply(&mut sol.state);
                        sol.answer.push(move_id.clone());
                    }
                }
                break;
            }
        }
    }
}

pub fn solve_subproblem00(
    solutions: &mut [TaskSolution],
    puzzle_info: &PuzzleType,
    squares: &[Vec<Vec<usize>>],
) {
    let interesting_positions = calc_positions(squares, 0, 0);
    let subspace = Subspace::new(interesting_positions);

    let moves = subspace.get_moves(puzzle_info, &[]);
    for sol in solutions.iter_mut() {
        let state: Vec<_> = subspace
            .interesting_positions
            .iter()
            .map(|&x| sol.task.solution_state[sol.state[x]])
            .collect();
        eprintln!("Before conv: {state:?}");
        // let state = sol.task.convert_state_to_colors(&state);
        let target_state = sol
            .task
            .convert_state_to_colors(&subspace.interesting_positions);
        eprintln!("Target state: {target_state:?}");
        // eprintln!("my state: {state:?}");

        for max_layers in 0.. {
            eprintln!("max_layers={max_layers}");
            let dist = |a: &[usize]| -> usize {
                for i in 0..a.len() {
                    if a[i] != target_state[i] {
                        return 1;
                    }
                }
                0
            };
            let mut moves_history = vec![];
            if dfs(
                max_layers,
                &mut state.clone(),
                &moves,
                &dist,
                &mut 0,
                &mut moves_history,
            ) {
                eprintln!("Found solution in {} moves!", moves_history.len());
                for &mov_id in moves_history.iter() {
                    for move_id in moves[mov_id].name.iter() {
                        let full_perm = &puzzle_info.moves[move_id];
                        full_perm.apply(&mut sol.state);
                        sol.answer.push(move_id.clone());
                    }
                }
                break;
            }
        }
    }
}

pub fn solve_nnn(data: &Data, task_type: &str, cube3_converter: &Cube3Converter, exact_perm: bool) {
    println!("Solving nnn: {task_type}");

    let mut solutions = TaskSolution::all_by_type(data, task_type, exact_perm);
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

    let show_ids = |a: &[usize]| {
        for line in [vec![0], vec![4, 1, 2, 3], vec![5]].iter() {
            let add_offset = || {
                if line.len() == 1 {
                    for _ in 0..sz + 2 {
                        eprint!(" ");
                    }
                }
            };
            let print_border = || {
                add_offset();
                for _ in 0..(sz + 2) * line.len() {
                    eprint!("-");
                }
                eprintln!();
            };
            print_border();
            for r in 0..sz {
                add_offset();
                for &sq_id in line.iter() {
                    eprint!("|");
                    for c in 0..sz {
                        let x = a.contains(&squares[sq_id][r][c]);
                        eprint!("{}", if x { "X" } else { "." });
                    }
                    eprint!("|");
                }
                eprintln!();
            }
            print_border();
        }
    };

    // 6 -> 32
    // let moves = ["r2", "-f6", "r0", "f6", "-r2", "-r0"];
    // let mut perm = Permutation::identity();

    // for mv in moves.iter() {
    //     perm = perm.combine(&puzzle_info.moves[&mv.to_string()]);
    // }
    // eprintln!("Perm: {:?}", perm);

    // for cycle in perm.cycles.iter() {
    //     eprintln!("Cycle: {cycle:?}");
    //     show_ids(cycle);
    // }

    // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, 0);
    // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, sz / 2);

    for d in 0..2 {
        // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, d);
        // for sol in solutions.iter() {
        //     sol.print(data);
        // }
    }
    if sz % 2 == 1 {
        solve_subproblem(&mut solutions, puzzle_info, &squares, 0, 0);
    }
    // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, 2);

    // for sol in solutions.iter() {
    //     sol.print(data);
    //     show_ids(&solutions[0].get_correct_colors_positions());
    // }

    let side_moves = get_cube_side_moves(sz);
    let mut it = 0;

    let mut hs = HashSet::new();
    let mut moves = vec![];

    let keys = puzzle_info.moves.keys().collect::<Vec<_>>();

    let mut rng = rand::thread_rng();

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

    for tr in par_maps.iter() {
        if hs.insert(tr.mv.permutation.cycles.clone()) {
            // eprintln!("Permutation: {:?}. {info:?}", mv.permutation);
            moves.push(tr.mv.clone());
        }
    }

    eprintln!("hm={}", hs.len());
    for sol in solutions.iter_mut() {
        // eprintln!("State: {:?}", sol.state);
        let need_moves = triangle_parity_solver(&sol.state, dsu.get_groups(), sol, sz);
        for mv in need_moves.iter() {
            // eprintln!("Need move: {:?}", mv.name);
            puzzle_info.moves[mv].apply(&mut sol.state);
            sol.answer.push(mv.to_string());
        }

        let mut need_to_do = vec![];

        let mut triangles_total_applied = 0;
        let mut triangles_groups_joined = 0;

        for triangles in triangles_by_dsu.iter() {
            if !triangles.is_empty() {
                let r = solve_triangle(&sol.state, triangles).unwrap();
                let mut to_do = vec![];
                for &id in r.iter().rev() {
                    to_do.push(triangles[id].clone());
                }
                need_to_do.push(to_do);
            }
        }

        // let mut all_triangles_to_apply: Vec<_> = need_to_do.iter().flatten().collect();
        // for i in 0..all_triangles_to_apply.len() {
        //     for j in i + 1..all_triangles_to_apply.len() {
        //         let tr1 = &all_triangles_to_apply[i];
        //         let tr2 = &all_triangles_to_apply[j];
        //         if tr1.can_combine(tr2, puzzle_info) {
        //             eprintln!("WOW {i} {j}");
        //         }
        //     }
        // }

        loop {
            let mut triangles_to_apply: Vec<Triangle> = vec![];
            for i in 0..need_to_do.len() {
                if need_to_do[i].is_empty() {
                    continue;
                }
                if !triangles_to_apply.is_empty()
                    && !triangles_to_apply[0]
                        .can_combine(need_to_do[i].last().unwrap(), puzzle_info)
                {
                    continue;
                }
                triangles_to_apply.push(need_to_do[i].pop().unwrap());
            }
            if triangles_to_apply.is_empty() {
                break;
            }
            let zz: Vec<_> = triangles_to_apply.iter().collect();
            if zz.len() > 1 {
                eprintln!("Combined {} triangles!", zz.len());
            }
            triangles_total_applied += zz.len();
            triangles_groups_joined += 1;
            let all_moves = Triangle::gen_combination_moves(&zz);
            for mv in all_moves.into_iter() {
                puzzle_info.moves[&mv].apply(&mut sol.state);
                sol.answer.push(mv.clone());
            }
        }

        eprintln!(
            "Task_id = {}. Used moves for now: {}",
            sol.task_id,
            sol.answer.len()
        );
        eprintln!("Triangles total: {triangles_total_applied}. Groups: {triangles_groups_joined}");

        // let mut all_masks: Vec<_> = (0..128usize).collect();
        // all_masks.sort_by_key(|m| m.count_ones());
        // for mask in all_masks.into_iter() {
        //     let mut all_ok = true;
        //     let mut state = sol.state.clone();
        //     for (i, c) in ["f1", "r1", "d1", "f0", "r0", "d0"].iter().enumerate() {
        //         if mask & (1 << i) != 0 {
        //             puzzle_info.moves[&c.to_string()].apply(&mut state);
        //         }
        //     }
        //     let mut zz = vec![];
        //     for triangles in triangles_by_dsu.iter() {
        //         if !triangles.is_empty() {
        //             zz.push(.is_none());
        //         }
        //     }

        //     eprintln!("{mask} -> {zz:?}");
        //     // if all_ok {
        //     //     eprintln!("All ok!");
        //     // } else {
        //     //     eprintln!("Not all ok!");
        //     // }
        // }
        // for side_mv_iter in 0..1 {
        //     loop {
        //         let mut changed = false;
        //         for delta in (1..=3).rev() {
        //             for mv in moves.iter() {
        //                 let mut cur_cnt_ok = 0;
        //                 for &cell_id in mv.permutation.cycles[0].iter() {
        //                     // if sol.task.solution_state[sol.state[cell_id]]
        //                     //     == sol.task.solution_state[cell_id]
        //                     // {
        //                     //     cur_cnt_ok += 1;
        //                     // }
        //                     if sol.state[cell_id] == cell_id {
        //                         cur_cnt_ok += 1;
        //                     }
        //                 }
        //                 mv.permutation.apply(&mut sol.state);
        //                 let mut next_cnt_ok = 0;
        //                 for &cell_id in mv.permutation.cycles[0].iter() {
        //                     if sol.state[cell_id] == cell_id {
        //                         next_cnt_ok += 1;
        //                     }
        //                 }
        //                 // eprintln!("cur_cnt_ok={cur_cnt_ok}, next_cnt_ok={next_cnt_ok}");
        //                 if next_cnt_ok >= cur_cnt_ok + delta {
        //                     changed = true;
        //                     // eprintln!("Changed!");
        //                     sol.answer.extend(mv.name.iter().cloned());
        //                     break;
        //                 } else {
        //                     mv.permutation.apply_rev(&mut sol.state);
        //                 }
        //             }
        //             if changed {
        //                 break;
        //             }
        //         }
        //         if !changed {
        //             break;
        //         }
        //     }
        //     // let side_mv = side_moves.choose(&mut rng).unwrap();
        //     // sol.answer.push(side_mv.to_string());
        //     // puzzle_info.moves[side_mv].apply(&mut sol.state);
        // }

        eprintln!("Before solving edges...");

        show_ids(&sol.get_correct_colors_positions());
        eprintln!("IDS:");
        show_ids(&sol.get_correct_positions());

        solve_edges(sol);

        sol.print(data);
        show_ids(&sol.get_correct_colors_positions());
        eprintln!("IDS:");
        show_ids(&sol.get_correct_positions());

        cube3_converter.solve(data, sol);
        sol.print(data);
        show_ids(&sol.get_correct_colors_positions());
        show_ids(&sol.get_correct_positions());
    }
}
