use std::{collections::BTreeSet, ops::Sub};

use crate::{
    data::Data, moves::SeveralMoves, permutation::Permutation, puzzle::Puzzle,
    puzzle_type::PuzzleType, sol_utils::TaskSolution,
};

fn build_square(sz: usize, offset: usize) -> Vec<Vec<usize>> {
    let mut res = vec![vec![0; sz]; sz];
    for r in 0..sz {
        for c in 0..sz {
            res[r][c] = offset + r * sz + c;
        }
    }
    res
}

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

pub fn solve_nnn(data: &Data, task_type: &str) {
    println!("Solving nnn: {task_type}");

    let mut solutions = TaskSolution::all_by_type(data, task_type);
    eprintln!("Tasks cnt: {}", solutions.len());
    solutions.truncate(1);
    let task_id = solutions[0].task_id;
    // eprintln!("Solving id={task_id}");

    let puzzle_info = data.puzzle_info.get(task_type).unwrap();

    let n = puzzle_info.n;
    eprintln!("n={n}");
    let sz = {
        let mut sz = 1;
        while 6 * sz * sz < n {
            sz += 1;
        }
        sz
    };
    eprintln!("sz={sz}");

    let squares: Vec<_> = (0..6)
        .map(|i| build_square(sz, i * sz * sz))
        .collect::<Vec<_>>();

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

    // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, 0);
    // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, sz / 2);

    for d in 0..2 {
        solve_subproblem(&mut solutions, puzzle_info, &squares, 0, d);
        // for sol in solutions.iter() {
        //     sol.print(data);
        // }
    }
    // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, 1);
    // solve_subproblem(&mut solutions, puzzle_info, &squares, 0, 2);

    for sol in solutions.iter() {
        sol.print(data);
        show_ids(&solutions[0].get_correct_colors_positions());
    }
}
