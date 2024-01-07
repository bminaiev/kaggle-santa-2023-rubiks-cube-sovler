use std::{
    collections::{hash_map::DefaultHasher, HashSet},
    hash::Hasher,
};

use crate::{
    data::Data,
    groups::{
        apply_precomputed_moves, apply_precomputed_moves_bfs, blocks_bfs, get_groups,
        precompute_moves, precompute_moves_from_final_state, PREC_LIMIT,
    },
    moves::{create_moves, SeveralMoves},
    puzzle::Puzzle,
    puzzle_type::PuzzleType,
    sol_utils::TaskSolution,
    utils::{get_blocks, get_blocks_by_several_moves},
};

// R2 == "r0x2"
// r = "r1"
// L = "r3"

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
            "d0", "d1x2", "d2x2", "d3", "f0", "f1x2", "f2x2", "f3", "r0", "r1", "r2", "r3",
        ],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &[
            "d0", "d1x2", "d2x2", "d3", "f0", "f1x2", "f2x2", "f3", "r0x2", "r1x2", "r2x2", "r3x2",
        ],
    ));
    move_groups.push(create_moves(
        puzzle_info,
        &[
            "d0", "d3", "f0x2", "f1x2", "f2x2", "f3x2", "r0x2", "r1x2", "r2x2", "r3x2",
        ],
    ));
    // Step 5 - Thistlethwaite #2
    move_groups.push(create_moves(
        puzzle_info,
        &["d0", "d3", "f0", "f3", "r0x2", "r3x2"],
    ));
    // Step 6 - Thistlethwaite #3
    move_groups.push(create_moves(
        puzzle_info,
        &["d0", "d3", "f0x2", "f3x2", "r0x2", "r3x2"],
    ));
    // Step 7 - Thistlethwaite #4
    move_groups.push(create_moves(
        puzzle_info,
        &["d0x2", "d3x2", "f0x2", "f3x2", "r0x2", "r3x2"],
    ));
    move_groups.push(create_moves(puzzle_info, &[]));
    move_groups
}

struct PermutationParity {
    mask: u64,
    res: u32,
}

impl PermutationParity {
    fn new() -> Self {
        Self { mask: 0, res: 0 }
    }

    fn add(&mut self, x: usize) {
        assert!((self.mask >> x) & 1 == 0);
        self.mask |= 1 << x;
        for i in 0..x {
            if (self.mask >> i) & 1 == 1 {
                self.res ^= 1;
            }
        }
    }
}

fn check_this_is_perm(a: &[usize]) {
    let mut b = a.to_vec();
    b.sort();
    b.dedup();
    assert!(b.len() == a.len());
}

fn calc_perm_parity(a: &[usize]) -> usize {
    check_this_is_perm(a);
    let mut res = 0;
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            res ^= (a[i] > a[j]) as usize;
        }
    }
    res
}

const LOW_EDGES: [[usize; 2]; 12] = [
    [1, 50],
    [8, 66],
    [7, 34],
    [39, 52],
    [72, 59],
    [81, 29],
    [14, 18],
    [87, 45],
    [88, 77],
    [71, 20],
    [94, 61],
    [40, 27],
];
const HIGH_EDGES: [[usize; 2]; 12] = [
    [2, 49],
    [4, 65],
    [11, 33],
    [43, 56],
    [68, 55],
    [82, 30],
    [13, 17],
    [91, 46],
    [84, 78],
    [75, 24],
    [93, 62],
    [36, 23],
];

fn analyze_state(
    prev_moves: &[SeveralMoves],
    next_moves: &[SeveralMoves],
    puzzle_info: &PuzzleType,
    task: &Puzzle,
    state: &[usize],
) {
    let moves = puzzle_info.get_all_moves();
    let blocks = get_blocks(puzzle_info.n, &moves);

    let mut id = vec![usize::MAX; puzzle_info.n];
    for (i, edge) in LOW_EDGES.iter().enumerate() {
        for &x in edge.iter() {
            id[x] = i;
        }
    }
    for (i, edge) in HIGH_EDGES.iter().enumerate() {
        for &x in edge.iter() {
            id[x] = i;
        }
    }

    let mut state = [
        48, 14, 13, 19, 68, 85, 86, 88, 40, 10, 89, 93, 95, 4, 39, 76, 47, 65, 52, 63, 45, 21, 57,
        33, 78, 54, 58, 29, 0, 34, 46, 67, 92, 62, 77, 32, 11, 69, 70, 8, 81, 38, 37, 75, 12, 20,
        49, 80, 15, 17, 18, 3, 66, 26, 25, 56, 24, 22, 53, 50, 79, 61, 30, 31, 35, 55, 27, 60, 43,
        42, 41, 87, 1, 73, 74, 84, 44, 59, 23, 51, 64, 7, 91, 16, 36, 6, 90, 71, 72, 5, 9, 2, 83,
        82, 94, 28,
    ];

    let groups = get_groups(&blocks, next_moves);

    let mut calc_hash = |a: &[usize], debug: bool| {
        if debug {
            eprintln!("State: {:?}", a);
        }
        let mut hasher = groups.hash(a);
        // let mut parity = 0;
        // for edges in [low_edges, high_edges].iter() {
        //     let mut perm = PermutationParity::new();
        //     for edge in edges.iter() {
        //         let id = id[a[edge[0]]];
        //         // hasher.write_usize(id);
        //         perm.add(id);
        //     }
        //     parity ^= perm.res;
        // }
        // hasher.write_u32(parity);
        hasher.finish()
    };
    let mut p1 = vec![];
    let mut p2 = vec![];
    for i in 0..LOW_EDGES.len() {
        let id1 = id[state[LOW_EDGES[i][0]]];
        let id2 = id[state[HIGH_EDGES[i][0]]];
        // eprintln!("{i}: {id1} {id2}");
        p1.push(id1);
        p2.push(id2);
    }
    eprintln!("PARITY1: {}", calc_perm_parity(&p1));
    eprintln!("PARITY2: {}", calc_perm_parity(&p2));
    eprintln!("HASH: {}", calc_hash(&state, true));
}

pub fn two_side(
    prev_moves: &[SeveralMoves],
    next_moves: &[SeveralMoves],
    puzzle_info: &PuzzleType,
    task: &Puzzle,
) {
    let moves = puzzle_info.get_all_moves();
    let blocks = get_blocks(puzzle_info.n, &moves);

    let mut id = vec![usize::MAX; puzzle_info.n];
    for (i, edge) in LOW_EDGES.iter().enumerate() {
        for &x in edge.iter() {
            id[x] = i;
        }
    }
    for (i, edge) in HIGH_EDGES.iter().enumerate() {
        for &x in edge.iter() {
            id[x] = i;
        }
    }

    let mut state = [
        48, 14, 13, 19, 68, 85, 86, 88, 40, 10, 89, 93, 95, 4, 39, 76, 47, 65, 52, 63, 45, 21, 57,
        33, 78, 54, 58, 29, 0, 34, 46, 67, 92, 62, 77, 32, 11, 69, 70, 8, 81, 38, 37, 75, 12, 20,
        49, 80, 15, 17, 18, 3, 66, 26, 25, 56, 24, 22, 53, 50, 79, 61, 30, 31, 35, 55, 27, 60, 43,
        42, 41, 87, 1, 73, 74, 84, 44, 59, 23, 51, 64, 7, 91, 16, 36, 6, 90, 71, 72, 5, 9, 2, 83,
        82, 94, 28,
    ];

    let groups = get_groups(&blocks, next_moves);

    let mut calc_hash = |a: &[usize], debug: bool| {
        if debug {
            eprintln!("State: {:?}", a);
        }
        let mut hasher = groups.hash(a);
        // let mut parity = 0;
        // for edges in [low_edges, high_edges].iter() {
        //     let mut perm = PermutationParity::new();
        //     for edge in edges.iter() {
        //         let id = id[a[edge[0]]];
        //         // hasher.write_usize(id);
        //         perm.add(id);
        //     }
        //     parity ^= perm.res;
        // }
        // hasher.write_u32(parity);
        hasher.finish()
    };
    let mut p1 = vec![];
    let mut p2 = vec![];
    for i in 0..LOW_EDGES.len() {
        let id1 = id[state[LOW_EDGES[i][0]]];
        let id2 = id[state[HIGH_EDGES[i][0]]];
        eprintln!("{i}: {id1} {id2}");
        p1.push(id1);
        p2.push(id2);
    }
    eprintln!("PARITY1: {}", calc_perm_parity(&p1));
    eprintln!("PARITY2: {}", calc_perm_parity(&p2));
    eprintln!("HASH: {}", calc_hash(&state, true));
    // eprintln!("START PRECALCULTATING...");
    // let prec = precompute_moves(puzzle_info.n, prev_moves, &mut calc_hash, 500_000);
    // eprintln!("PRECALC FINISHED!");
    // let mut answer = vec![];
    // assert!(apply_precomputed_moves_bfs(
    //     &mut state,
    //     &prec,
    //     calc_hash,
    //     &mut answer,
    //     prev_moves
    // ));

    eprintln!(
        "NEW STATE: {}",
        task.convert_state(
            &state
                .iter()
                .map(|x| task.solution_state[*x])
                .collect::<Vec<_>>()
        )
    )
}

fn convert_to_low_high_edges(a: &[usize]) -> [usize; 2] {
    let mut res = [usize::MAX; 2];
    for (idx, edges) in [LOW_EDGES, HIGH_EDGES].iter().enumerate() {
        for (pos, edge) in edges.iter().enumerate() {
            if edge.iter().all(|x| a.contains(x)) {
                res[idx] = pos;
            }
        }
    }
    res
}

fn conv_lh(lh: &[usize]) -> usize {
    lh[0] * 12 + lh[1]
}

fn dfs(
    more_layers: usize,
    state: &mut [usize],
    moves: &[SeveralMoves],
    estimate_dist: &impl Fn(&[usize]) -> usize,
) -> bool {
    let dist = estimate_dist(state);
    if dist == 0 {
        return true;
    }
    if dist > more_layers {
        return false;
    }
    for mov in moves.iter() {
        if mov.name.len() > more_layers {
            continue;
        }
        mov.permutation.apply(state);
        if dfs(more_layers - mov.name.len(), state, moves, estimate_dist) {
            return true;
        }
        mov.permutation.apply_rev(state);
    }
    false
}

pub fn step_3(
    prev_moves: &[SeveralMoves],
    next_moves: &[SeveralMoves],
    puzzle_info: &PuzzleType,
    task: &Puzzle,
) {
    let moves = puzzle_info.get_all_moves();
    let blocks = get_blocks(puzzle_info.n, &moves);
    let new_blocks = get_blocks_by_several_moves(puzzle_info.n, next_moves);

    // let groups = get_groups(&blocks, next_moves);

    let mut stay_together = vec![];
    for bl in new_blocks.iter() {
        if !blocks.contains(bl) {
            eprintln!("New block: {:?}", bl);
            if bl.len() == 4 {
                // TODO: remove this if
                // stay_together.push(bl.clone());
                let lh = convert_to_low_high_edges(bl);
                // eprintln!("LOW/HIGH: {:?}", convert_to_low_high_edges(bl));
                stay_together.push([
                    LOW_EDGES[lh[0]][0],
                    LOW_EDGES[lh[0]][1],
                    HIGH_EDGES[lh[1]][0],
                    HIGH_EDGES[lh[1]][1],
                ]);
                // let cur_group = *groups.by_elem.get(bl).unwrap();
                // eprintln!("CUR GROUP: {:?}", groups.groups[cur_group]);
            }
        }
    }

    let mut all_keys = HashSet::new();

    // [l/h id][stay_together_id] -> min len
    let mut cost = vec![vec![usize::MAX; 4]; 144];

    for (i, stay) in stay_together.iter().enumerate() {
        eprintln!("Stay together {}: {:?}", i, stay);
        let positions = blocks_bfs(stay, prev_moves);
        for (k, v) in positions.iter() {
            let lh = convert_to_low_high_edges(k);
            let key = conv_lh(&lh);
            cost[key][i] = *v;
            all_keys.insert(k.clone());
        }
    }

    let mut id = vec![usize::MAX; puzzle_info.n];
    for (i, edge) in LOW_EDGES.iter().enumerate() {
        for &x in edge.iter() {
            id[x] = i;
        }
    }
    for (i, edge) in HIGH_EDGES.iter().enumerate() {
        for &x in edge.iter() {
            id[x] = i;
        }
    }

    let mut state = [
        48, 14, 13, 19, 68, 85, 86, 88, 40, 10, 89, 93, 95, 4, 39, 76, 47, 65, 52, 63, 45, 21, 57,
        33, 78, 54, 58, 29, 0, 34, 46, 67, 92, 62, 77, 32, 11, 69, 70, 8, 81, 38, 37, 75, 12, 20,
        49, 80, 15, 17, 18, 3, 66, 26, 25, 56, 24, 22, 53, 50, 79, 61, 30, 31, 35, 55, 27, 60, 43,
        42, 41, 87, 1, 73, 74, 84, 44, 59, 23, 51, 64, 7, 91, 16, 36, 6, 90, 71, 72, 5, 9, 2, 83,
        82, 94, 28,
    ];

    // dist is at least this big
    let esimate_dist = |a: &[usize]| -> usize {
        let mut pos = [[usize::MAX; 2]; 12];
        for i in 0..LOW_EDGES.len() {
            let who = id[a[LOW_EDGES[i][0]]];
            pos[who][0] = i;
            let who = id[a[HIGH_EDGES[i][0]]];
            pos[who][1] = i;
        }
        let mut mincost = [usize::MAX; 4];
        for i in 0..pos.len() {
            let lh = conv_lh(&pos[i]);
            for j in 0..4 {
                mincost[j] = mincost[j].min(cost[lh][j]);
            }
        }
        *mincost.iter().max().unwrap()
    };
    let at_least_dist = esimate_dist(&state);
    eprintln!("AT LEAST DIST: {at_least_dist}");

    for maxdist in 1.. {
        eprintln!("CUR MAXDIST: {maxdist}");
        let mut now_state = state;
        if dfs(maxdist, &mut now_state, prev_moves, &esimate_dist) {
            eprintln!("FOUND!");
            state = now_state;
            break;
        }
    }

    let groups = get_groups(&blocks, next_moves);

    let mut calc_hash = |a: &[usize], debug: bool| {
        if debug {
            eprintln!("State: {:?}", a);
        }
        let mut hasher = groups.hash(a);
        // let mut parity = 0;
        // for edges in [low_edges, high_edges].iter() {
        //     let mut perm = PermutationParity::new();
        //     for edge in edges.iter() {
        //         let id = id[a[edge[0]]];
        //         // hasher.write_usize(id);
        //         perm.add(id);
        //     }
        //     parity ^= perm.res;
        // }
        // hasher.write_u32(parity);
        hasher.finish()
    };
    let mut p1 = vec![];
    let mut p2 = vec![];
    for i in 0..LOW_EDGES.len() {
        let id1 = id[state[LOW_EDGES[i][0]]];
        let id2 = id[state[HIGH_EDGES[i][0]]];
        eprintln!("{i}: {id1} {id2}");
        p1.push(id1);
        p2.push(id2);
    }
    eprintln!("PARITY1: {}", calc_perm_parity(&p1));
    eprintln!("PARITY2: {}", calc_perm_parity(&p2));
    eprintln!("HASH: {}", calc_hash(&state, true));
    // eprintln!("START PRECALCULTATING...");
    // let prec = precompute_moves(puzzle_info.n, prev_moves, &mut calc_hash, 500_000);
    // eprintln!("PRECALC FINISHED!");
    // let mut answer = vec![];
    // assert!(apply_precomputed_moves_bfs(
    //     &mut state,
    //     &prec,
    //     calc_hash,
    //     &mut answer,
    //     prev_moves
    // ));

    eprintln!(
        "NEW STATE: {}",
        task.convert_state(
            &state
                .iter()
                .map(|x| task.solution_state[*x])
                .collect::<Vec<_>>()
        )
    )
}

pub fn solve4(data: &Data, task_type: &str) {
    let mut solutions = TaskSolution::all_by_type(data, task_type);
    eprintln!("Tasks cnt: {}", solutions.len());
    solutions.truncate(1);
    let task_id = solutions[0].task_id;
    // eprintln!("Solving id={task_id}");

    let puzzle_info = data.puzzle_info.get(task_type).unwrap();

    let move_groups = create_move_groups(puzzle_info);
    eprintln!("Total move groups: {}", move_groups.len());

    let moves = puzzle_info.get_all_moves();
    let blocks = get_blocks(puzzle_info.n, &moves);

    if true {
        step_3(
            &move_groups[2],
            &move_groups[3],
            puzzle_info,
            &data.puzzles[task_id],
        );
    } else {
        for step in 0..2 {
            eprintln!("Step: {step}");
            let groups = get_groups(&blocks, &move_groups[step + 1]);
            eprintln!("Groups are calculated.");
            for (i, group) in groups.groups.iter().enumerate() {
                eprintln!("Group {i}: {:?}", group);
            }
            let mut calc_hash = |a: &[usize], debug: bool| groups.hash(a).finish();
            let prec = precompute_moves(puzzle_info.n, &move_groups[step], &mut calc_hash, 500_000);
            for sol in solutions.iter_mut() {
                if sol.failed_on_stage.is_some() {
                    continue;
                }
                if !apply_precomputed_moves_bfs(
                    &mut sol.state,
                    &prec,
                    calc_hash,
                    &mut sol.answer,
                    &move_groups[step],
                ) {
                    sol.failed_on_stage = Some(step);
                    eprintln!("FAILED!");
                } else {
                    eprintln!("CURRENT STATE: {:?}", sol.state);
                    sol.print(data);
                    if step == 1 {
                        analyze_state(
                            &move_groups[1],
                            &move_groups[2],
                            puzzle_info,
                            &data.puzzles[sol.task_id],
                            &sol.state,
                        );
                    }
                }
            }
        }
        for sol in solutions.iter() {
            eprintln!("ANALYZE {}", sol.task_id);
            analyze_state(
                &move_groups[1],
                &move_groups[2],
                puzzle_info,
                &data.puzzles[sol.task_id],
                &sol.state,
            );
            // sol.print(data);
        }
    }
}
