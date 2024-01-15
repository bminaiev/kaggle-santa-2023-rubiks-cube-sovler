use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    hash::Hasher,
    io::Read,
};

use crate::{
    checker::check_solution,
    data::Data,
    groups::{apply_precomputed_moves, get_groups, precompute_moves, Edge, Groups, PREC_LIMIT},
    moves::{create_moves, SeveralMoves},
    puzzle_type::PuzzleType,
    rotations::{apply_rotation, apply_rotations, get_rotations_dists},
    sol_utils::TaskSolution,
    utils::{get_all_perms, get_blocks, get_start_permutation},
};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::io::Write;

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

pub struct Solver3 {
    move_groups: Vec<Vec<SeveralMoves>>,
    groups: Vec<Groups>,
    blocks: Vec<Vec<usize>>,
    precalcs: Vec<HashMap<u64, Edge>>,
}

impl Solver3 {
    pub fn new_fake(data: &Data) -> Self {
        Self {
            move_groups: vec![],
            groups: vec![],
            blocks: vec![],
            precalcs: vec![],
        }
    }

    pub fn new(data: &Data) -> Self {
        let puzzle_info = data.puzzle_info.get("cube_3/3/3").unwrap();
        let move_groups = create_move_groups(puzzle_info);
        eprintln!("Total move groups: {}", move_groups.len());
        let moves = puzzle_info.get_all_moves();
        let blocks = get_blocks(puzzle_info.n, &moves);
        let groups: Vec<_> = move_groups.iter().map(|m| get_groups(&blocks, m)).collect();

        let mut res = Self {
            move_groups,
            groups,
            blocks,
            precalcs: vec![],
        };

        // create precals folder if not exists
        std::fs::create_dir_all("precalcs").unwrap();

        let precalcs: Vec<_> = (0..res.move_groups.len() - 1)
            .into_par_iter()
            .map(|step| {
                let precalc_file = format!("precalcs/{step}.txt");
                if std::path::Path::new(&precalc_file).exists() {
                    eprintln!("Loading precalc from {}", precalc_file);
                    let mut precalc = HashMap::new();
                    let mut f = std::fs::File::open(precalc_file).unwrap();
                    let mut buf = String::new();
                    f.read_to_string(&mut buf).unwrap();
                    for line in buf.lines() {
                        let mut it = line.split(' ');
                        let hash = it.next().unwrap().parse::<u64>().unwrap();
                        let next_state_hash = it.next().unwrap().parse::<u64>().unwrap();
                        let mov_idx = it.next().unwrap().parse::<usize>().unwrap();
                        let len = it.next().unwrap().parse::<usize>().unwrap();
                        precalc.insert(
                            hash,
                            Edge {
                                next_state_hash,
                                mov_idx,
                                len,
                            },
                        );
                    }
                    return precalc;
                }

                let mut calc_hash = |a: &[usize], _debug: bool| res.calc_hash(step, a);
                let prec = precompute_moves(
                    puzzle_info.n,
                    &res.move_groups[step],
                    &mut calc_hash,
                    PREC_LIMIT,
                );

                {
                    // save precalc
                    let mut f = std::fs::File::create(precalc_file).unwrap();
                    for (hash, edge) in prec.iter() {
                        writeln!(
                            f,
                            "{} {} {} {}",
                            hash, edge.next_state_hash, edge.mov_idx, edge.len
                        )
                        .unwrap();
                    }
                }

                prec
            })
            .collect();

        res.precalcs = precalcs;
        res
    }

    fn calc_hash(&self, step: usize, a: &[usize]) -> u64 {
        let mut hasher = self.groups[step + 1].hash(a);

        let hacks_info = step == 3;
        if hacks_info {
            for block in self.blocks.iter() {
                if block.len() != 3 {
                    continue;
                }
                let min = block.iter().map(|&x| a[x]).min().unwrap();
                hasher.write_usize(min);
            }
        }
        hasher.finish()
    }

    pub fn solve_task(&self, task: &mut TaskSolution) {
        for step in 0..self.precalcs.len() {
            let res = apply_precomputed_moves(
                &mut task.state,
                &self.precalcs[step],
                |a, _debug| self.calc_hash(step, a),
                &mut task.answer,
                &self.move_groups[step],
            );
            assert!(res);
        }
    }

    fn dfs(
        &self,
        more_moves: usize,
        rotation_dists: &[usize],
        rot: usize,
        state: &mut [usize],
        answer: &mut Vec<String>,
        step: usize,
    ) -> bool {
        let edge = self.precalcs[step]
            .get(&self.calc_hash(step, state))
            .unwrap();
        if edge.len > more_moves || rotation_dists[rot] > more_moves {
            return false;
        }
        if edge.len == 0 && rotation_dists[rot] == 0 {
            eprintln!("Solution finished with rot: {rot}");
            return true;
        }
        for mv in self.move_groups[step].iter() {
            if more_moves < mv.name.len() {
                continue;
            }
            mv.permutation.apply(state);
            let new_rot = apply_rotations(rot, mv);
            answer.extend(mv.name.clone());
            if self.dfs(
                more_moves - mv.name.len(),
                rotation_dists,
                new_rot,
                state,
                answer,
                step,
            ) {
                return true;
            }
            answer.truncate(answer.len() - mv.name.len());
            mv.permutation.apply_rev(state);
        }

        false
    }

    pub fn solve_task_with_rotations(&self, task: &mut TaskSolution) {
        let mut rot = 0;
        for step in 0..self.precalcs.len() {
            let rotation_dists =
                get_rotations_dists(&self.move_groups[step], &self.move_groups[step + 1]);
            for sol_len in 0.. {
                eprintln!("Trying len {sol_len}...");
                let mut answer = vec![];
                let mut state = task.state.clone();
                if self.dfs(sol_len, &rotation_dists, rot, &mut state, &mut answer, step) {
                    eprintln!("Found solution for step {step}!");
                    for mv in answer.iter() {
                        task.answer.push(mv.clone());
                        task.task.info.moves[mv].apply(&mut task.state);
                        rot = apply_rotation(rot, mv);
                    }
                    break;
                }
            }
        }
        assert_eq!(rot, 0);
    }
}

pub fn solve3(data: &Data, task_type: &str) {
    eprintln!("SOLVING {task_type}");
    let mut solutions = TaskSolution::all_by_type(data, task_type);

    let solver = Solver3::new(data);

    for sol in solutions.iter_mut() {
        solver.solve_task(sol);
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
