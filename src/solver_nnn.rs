use std::collections::{BTreeSet, HashMap, HashSet};

use rand::{rngs::StdRng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    cube_edges_calculator::{build_squares, calc_cube_centers, calc_cube_edges, calc_edges_score},
    data::Data,
    dsu::Dsu,
    edge_solver::solve_edges,
    edge_solver_dwalton::solve_edges_dwalton,
    greedy::greedy_cube_optimizer,
    moves::SeveralMoves,
    parallel_triangle_solver::{solve_all_triangles, solve_all_triangles_greedy},
    permutation::Permutation,
    puzzle_type::PuzzleType,
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
    to_cube3_converter::Cube3Converter,
    triangle_solver::Triangle,
    triangles_parity::triangle_parity_solver,
    utils::{
        calc_cube_side_size, conv_cube_to_dwalton, conv_dwalton_moves, get_cube_side_moves,
        show_cube_ids,
    },
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
            .map(|&x| sol.state[x])
            .collect();
        eprintln!("Before conv: {state:?}");
        // let state = sol.task.convert_state_to_colors(&state);
        let target_state: Vec<_> = subspace
            .interesting_positions
            .iter()
            .map(|&pos| sol.target_state[pos])
            .collect();
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

#[derive(Debug)]
enum SolutionStage {
    Empty,
    CentersDone,
    EdgesDone,
}

fn strip_bad_suffix(sol: &mut TaskSolution, data: &Data, just_centers: bool) -> SolutionStage {
    let old_moves = sol.answer.clone();
    sol.reset(data);

    let n = sol.task.info.n;
    let sz = calc_cube_side_size(n);
    let squares = build_squares(sz);
    let edges = calc_cube_edges(&squares);
    let cube_centers = calc_cube_centers(&squares);

    let mut first_centers_correct = usize::MAX;

    for mv in old_moves.iter() {
        sol.append_move(mv);
        let centers_correct =
            (0..n).all(|i| !cube_centers[i] || sol.state[i] == sol.target_state[i]);
        if !centers_correct {
            continue;
        }
        if first_centers_correct == usize::MAX {
            first_centers_correct = sol.answer.len();
            if just_centers {
                break;
            }
        }
        let edges_score = calc_edges_score(&edges, &sol.state, &sol.target_state);
        let edges_correct = (0..edges_score.len()).all(|i| edges_score[i] == edges[i].len());
        if edges_correct {
            return SolutionStage::EdgesDone;
        }
    }
    sol.reset(data);
    if first_centers_correct == usize::MAX {
        return SolutionStage::Empty;
    }
    for mv in old_moves.into_iter().take(first_centers_correct) {
        sol.append_move(&mv);
    }
    SolutionStage::CentersDone
}

pub fn fix_permutations_in_log(
    data: &Data,
    task_type: &str,
    log: &mut SolutionsLog,
    cube3_converter: &Cube3Converter,
) {
    let mut last_event = HashMap::new();
    for (i, event) in log.events.iter().enumerate() {
        last_event.insert(event.task_id, i);
    }
    const REDO_EDGES: bool = true;
    for i in 0..log.events.len() {
        let event = &log.events[i];
        if last_event[&event.task_id] != i {
            continue;
        }
        let mut task = TaskSolution::new(data, event.task_id);
        if task.task.puzzle_type != task_type {
            continue;
        }
        if task.task_id != 281 {
            continue;
        }
        eprintln!(
            "Checking solution for task {}. Len: {}. Type: {}",
            event.task_id,
            event.solution.len(),
            task.task.get_color_type()
        );
        for mv in event.solution.iter() {
            task.append_move(mv);
        }

        let n = task.task.info.n;
        let sz = calc_cube_side_size(n);
        let correct_positions = task.get_correct_colors_positions();
        if correct_positions.len() == n && !REDO_EDGES {
            eprintln!("WOW! Correct solution!");
            continue;
        }
        eprintln!("Task type: {}", task.task.get_color_type());
        show_cube_ids(&correct_positions, sz);
        // TODO: change?
        let stage = strip_bad_suffix(&mut task, data, REDO_EDGES);
        eprintln!("Stage: {:?}", stage);
        show_cube_ids(&task.get_correct_colors_positions(), sz);
        eprintln!("Current sol len: {}", task.answer.len());
        match stage {
            SolutionStage::Empty => unreachable!(),
            SolutionStage::CentersDone => {
                solve_edges(&mut task);
                show_cube_ids(&task.get_correct_colors_positions(), sz);
                eprintln!("EDGES SOLVED.. SAVE PROGRESS!");
                log.append(&task);
            }
            SolutionStage::EdgesDone => {}
        }
        let exact_perm = task.task.need_exact_perm();
        cube3_converter.solve(data, &mut task, exact_perm);
        task.print(data);
        show_cube_ids(&task.get_correct_colors_positions(), sz);
        if task.is_solved() {
            eprintln!("WOW! Solved!");
            log.append(&task);
        } else {
            eprintln!("HMMM??? NOT SOLVED? WHY???");
        }
        // break;
    }
}

pub fn solve_nnn(
    data: &Data,
    task_type: &str,
    cube3_converter: &Cube3Converter,
    exact_perm: bool,
    log: &mut SolutionsLog,
) {
    println!("Solving nnn: {task_type}");

    let mut solutions = TaskSolution::all_by_type(data, task_type, exact_perm);
    // let mut solutions: Vec<_> = solutions
    //     .into_iter()
    //     .filter(|t| t.task.get_color_type() == "B")
    //     .collect();
    // solutions.reverse();
    eprintln!("Tasks cnt: {}", solutions.len());
    solutions.swap(0, 3);
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
    let mut it = 0;

    let mut hs = HashSet::new();
    let mut moves = vec![];

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

    for tr in par_maps.iter() {
        if hs.insert(tr.mv.permutation.cycles.clone()) {
            // eprintln!("Permutation: {:?}. {info:?}", mv.permutation);
            moves.push(tr.mv.clone());
        }
    }

    let triangle_groups: Vec<_> = triangles_by_dsu
        .into_iter()
        .filter(|x| !x.is_empty())
        .collect();

    eprintln!("hm={}", hs.len());

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
        if exact_perm {
            eprintln!("Solving parity...");
            let need_moves = triangle_parity_solver(&sol.state, dsu.get_groups(), sol, sz, false);
            for mv in need_moves.iter() {
                // eprintln!("Need move: {:?}", mv.name);
                puzzle_info.moves[mv].apply(&mut sol.state);
                sol.answer.push(mv.to_string());
            }
        } else if sz > 5 {
            greedy_cube_optimizer(sol);
        }

        if sz % 2 == 0 {
            loop {
                let mut sol_tmp = sol.clone();
                sol_tmp.make_random_moves(&mut rng);
                let with_moves = sol_tmp.clone();
                solve_all_triangles_greedy(&triangle_groups, &mut sol_tmp, exact_perm);
                eprintln!("After solving triangles greedily...");
                show_ids(&sol_tmp.get_correct_colors_positions());
                if !solve_edges(&mut sol_tmp) {
                    eprintln!("FAILED TO SOLVE EDGES... Need to change something?");
                } else if cube3_converter.solve(data, &mut sol_tmp, exact_perm) {
                    eprintln!("FOUND GOOD RANDOM MOVES!");
                    *sol = with_moves;
                    break;
                } else {
                    eprintln!("FAILED TO SOVLE 3x3");
                }
            }
        }

        solve_all_triangles(&triangle_groups, sol, exact_perm);

        eprintln!("Before solving edges...");

        show_ids(&sol.get_correct_colors_positions());

        solve_edges(sol);

        sol.print(data);
        show_ids(&sol.get_correct_colors_positions());

        cube3_converter.solve(data, sol, exact_perm);
        sol.print(data);
        show_ids(&sol.get_correct_colors_positions());

        log.append(sol);
    }
}
