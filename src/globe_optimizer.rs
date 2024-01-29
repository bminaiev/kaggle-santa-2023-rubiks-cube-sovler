use rand::{rngs::StdRng, SeedableRng};

use crate::{
    data::Data,
    globe_jaapsch::{globe_final_rows_move, GlobeState},
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
};

#[derive(Clone)]
struct Step {
    up_delta: i32,
    down_delta: i32,
    rotate_pos: Option<i32>,
    inside_up_delta: i32,
    inside_down_delta: i32,
}

struct SolutionLayer {
    steps: Vec<Step>,
}

impl SolutionLayer {
    fn from_moves(moves: &[GlobeMove], row1: usize) -> Option<Self> {
        let mut steps = vec![];
        let mut up_delta = 0i32;
        let mut down_delta = 0i32;
        let mut iter = 0;
        while iter != moves.len() {
            let mv = &moves[iter];
            match *mv {
                GlobeMove::Rotate(col) => {
                    if up_delta.signum() == down_delta.signum() && up_delta != 0 {
                        eprintln!("WTF? {up_delta} {down_delta}");
                        unreachable!()
                    }
                    let mut inside_up_delta = 0;
                    let mut inside_down_delta = 0;
                    loop {
                        iter += 1;
                        match moves[iter] {
                            GlobeMove::Rotate(col2) => {
                                if col != col2 {
                                    eprintln!("BAD SOLUTION!");
                                    return None;
                                }
                                break;
                            }
                            GlobeMove::RowShift { row, delta } => {
                                if row == row1 {
                                    inside_up_delta += delta;
                                } else {
                                    inside_down_delta += delta;
                                }
                            }
                        }
                    }
                    steps.push(Step {
                        up_delta,
                        down_delta,
                        rotate_pos: Some(col),
                        inside_up_delta,
                        inside_down_delta,
                    });
                    up_delta = 0;
                    down_delta = 0;
                }
                GlobeMove::RowShift { row, delta } => {
                    if row == row1 {
                        up_delta += delta;
                    } else {
                        down_delta += delta;
                    }
                }
            }
            iter += 1;
        }
        if up_delta != 0 || down_delta != 0 {
            steps.push(Step {
                up_delta,
                down_delta,
                rotate_pos: None,
                inside_up_delta: 0,
                inside_down_delta: 0,
            });
        }
        Some(Self { steps })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GlobeMove {
    Rotate(i32),
    RowShift { row: usize, delta: i32 },
}

impl GlobeMove {
    fn from_str(s: &str) -> GlobeMove {
        if let Some(col) = s.strip_prefix('f') {
            GlobeMove::Rotate(col.parse().unwrap())
        } else if let Some(col) = s.strip_prefix("-f") {
            GlobeMove::Rotate(col.parse().unwrap())
        } else if let Some(col) = s.strip_prefix('r') {
            GlobeMove::RowShift {
                row: col.parse().unwrap(),
                delta: -1,
            }
        } else if let Some(col) = s.strip_prefix("-r") {
            GlobeMove::RowShift {
                row: col.parse().unwrap(),
                delta: 1,
            }
        } else {
            eprintln!("Unknown move: {}", s);
            unreachable!()
        }
    }

    pub fn renumerate_rows(&mut self, row1: usize, row2: usize) {
        match self {
            GlobeMove::Rotate(_) => {}
            GlobeMove::RowShift { row, delta: _ } => *row = if *row == 0 { row1 } else { row2 },
        }
    }

    pub fn to_strings(&self) -> Vec<String> {
        match self {
            GlobeMove::Rotate(col) => vec![format!("f{}", col)],
            GlobeMove::RowShift { row, delta } => {
                if *delta > 0 {
                    vec![format!("-r{row}"); delta.abs() as usize]
                } else {
                    vec![format!("r{row}"); delta.abs() as usize]
                }
            }
        }
    }
}

fn parse_existing_solution(moves: &[String], state: &GlobeState) -> Option<Vec<SolutionLayer>> {
    let mut res = vec![];
    let mut sum_related_rows = 0;
    for row1 in 0..(state.n_rows + 1) / 2 {
        let row2 = state.n_rows - 1 - row1;
        let mut related_moves = vec![];
        for mv in moves.iter() {
            let mv = GlobeMove::from_str(mv);
            match mv {
                GlobeMove::Rotate(_col) => {
                    if related_moves.last() == Some(&mv) {
                        related_moves.pop();
                    } else {
                        related_moves.push(mv);
                    }
                }
                GlobeMove::RowShift { row, .. } => {
                    if row == row1 || row == row2 {
                        related_moves.push(mv);
                    }
                }
            }
        }
        sum_related_rows += related_moves.len();
        res.push(SolutionLayer::from_moves(&related_moves, row1)?);
    }
    eprintln!("Sum related: {}", sum_related_rows);
    Some(res)
}

#[derive(Default)]
struct PossibleOffset {
    left: i32,
    right: i32,
}

impl PossibleOffset {
    fn add(&mut self, up_delta: i32, down_delta: i32) {
        for &delta in [up_delta, down_delta].iter() {
            if delta > 0 {
                self.left += delta;
            } else {
                self.right += delta.abs();
            }
        }
    }
}

#[derive(Debug)]
struct Join {
    layer: usize,
    idx: usize,
    offset: i32,
}

fn combine_layers(layers: &[SolutionLayer], state: &GlobeState) -> Vec<GlobeMove> {
    let mut iters = vec![0; layers.len()];
    let lens: Vec<_> = layers.iter().map(|x| x.steps.len()).collect();
    eprintln!("Lens: {:?}", lens);
    let mut cur_delta = vec![0; layers.len()];
    let mut total_joined = 0;
    let mut res_moves = vec![];
    loop {
        let mut found = None;
        for max_skip in 0..50 {
            if found.is_some() {
                break;
            }
            for l1 in 0..iters.len() {
                for l2 in l1 + 1..iters.len() {
                    for i1 in iters[l1]..(iters[l1] + max_skip + 1).min(layers[l1].steps.len()) {
                        for i2 in iters[l2]..(iters[l2] + max_skip + 1).min(layers[l2].steps.len())
                        {
                            if let Some(rot1) = layers[l1].steps[i1].rotate_pos {
                                if let Some(rot2) = layers[l2].steps[i2].rotate_pos {
                                    let mut offset1 = PossibleOffset::default();
                                    for mv in layers[l1].steps[iters[l1]..=i1].iter() {
                                        offset1.add(mv.up_delta, mv.down_delta);
                                    }
                                    let mut offset2 = PossibleOffset::default();
                                    for mv in layers[l2].steps[iters[l2]..=i2].iter() {
                                        offset2.add(mv.up_delta, mv.down_delta);
                                    }
                                    let base_pos1 = cur_delta[l1] + rot1;
                                    let mut target1 = vec![i32::MAX; state.n_cols];
                                    for offset in -offset1.left..=offset1.right {
                                        let pos = state.calc_col(base_pos1 + offset);
                                        target1[pos as usize] = offset;
                                    }
                                    let base_pos2 = cur_delta[l2] + rot2;
                                    let mut target2 = vec![i32::MAX; state.n_cols];
                                    for offset in -offset2.left..=offset2.right {
                                        let pos = state.calc_col(base_pos2 + offset);
                                        target2[pos as usize] = offset;
                                    }
                                    for x in 0..target1.len() {
                                        if target1[x] != i32::MAX && target2[x] != i32::MAX {
                                            found = Some([
                                                Join {
                                                    layer: l1,
                                                    idx: i1,
                                                    offset: target1[x],
                                                },
                                                Join {
                                                    layer: l2,
                                                    idx: i2,
                                                    offset: target2[x],
                                                },
                                            ]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(mut join) = found {
            let mut last_moves = vec![];
            for join in join.iter_mut() {
                let layer = join.layer;
                let idx = join.idx;
                let row1 = layer;
                let row2 = state.n_rows - 1 - row1;
                for cur_idx in iters[layer]..=idx {
                    let change = &layers[layer].steps[cur_idx];
                    for _cnt in 0..change.up_delta.abs() {
                        if join.offset.signum() != change.up_delta.signum() && join.offset != 0 {
                            cur_delta[layer] += join.offset.signum();
                            join.offset -= join.offset.signum();
                            res_moves.push(GlobeMove::RowShift {
                                row: row2,
                                delta: -change.up_delta.signum(),
                            });
                        } else {
                            res_moves.push(GlobeMove::RowShift {
                                row: row1,
                                delta: change.up_delta.signum(),
                            });
                        }
                    }
                    for _cnt in 0..change.down_delta.abs() {
                        if join.offset.signum() != change.down_delta.signum() && join.offset != 0 {
                            cur_delta[layer] += join.offset.signum();
                            join.offset -= join.offset.signum();
                            res_moves.push(GlobeMove::RowShift {
                                row: row1,
                                delta: -change.down_delta.signum(),
                            });
                        } else {
                            res_moves.push(GlobeMove::RowShift {
                                row: row2,
                                delta: change.down_delta.signum(),
                            });
                        }
                    }
                    let rotate = GlobeMove::Rotate(
                        state.calc_col(change.rotate_pos.unwrap() + cur_delta[layer]),
                    );
                    if cur_idx == idx {
                        last_moves.push(change);
                    } else {
                        res_moves.push(rotate);
                        apply_row_moves(
                            &mut res_moves,
                            change.inside_up_delta,
                            change.inside_down_delta,
                            row1,
                            row2,
                        );
                        res_moves.push(rotate);
                    }
                }
                assert_eq!(join.offset, 0);
                iters[join.layer] = join.idx + 1;
                // if let Some(cur_last_rotate) = last_rotate {
                //     assert_eq!(cur_last_rotate, res_moves.last().unwrap().clone());
                // } else {
                //     last_rotate = res_moves.pop();
                // }
            }
            {
                let rot_pos1 =
                    state.calc_col(last_moves[0].rotate_pos.unwrap() + cur_delta[join[0].layer]);
                let rot_pos2 =
                    state.calc_col(last_moves[1].rotate_pos.unwrap() + cur_delta[join[1].layer]);
                assert_eq!(rot_pos1, rot_pos2);
                res_moves.push(GlobeMove::Rotate(rot_pos1));
                for (lm, row) in last_moves.iter().zip([join[0].layer, join[1].layer].iter()) {
                    apply_row_moves(
                        &mut res_moves,
                        lm.inside_up_delta,
                        lm.inside_down_delta,
                        *row,
                        state.n_rows - 1 - *row,
                    );
                }
                res_moves.push(GlobeMove::Rotate(rot_pos1));
            }
            total_joined += 1;
        } else {
            let mut ids: Vec<_> = (0..layers.len()).collect();
            ids.sort_by_key(|&x| iters[x]);
            let mut found = false;
            for &id in ids.iter() {
                if iters[id] != layers[id].steps.len() {
                    let step = &layers[id].steps[iters[id]];
                    let row1 = id;
                    let row2 = state.n_rows - 1 - id;
                    apply_row_moves(&mut res_moves, step.up_delta, step.down_delta, row1, row2);
                    if let Some(pos) = step.rotate_pos {
                        let real_pos = state.calc_col(cur_delta[id] + pos);
                        res_moves.push(GlobeMove::Rotate(real_pos));
                    }
                    apply_row_moves(
                        &mut res_moves,
                        step.inside_up_delta,
                        step.inside_down_delta,
                        row1,
                        row2,
                    );
                    if let Some(pos) = step.rotate_pos {
                        let real_pos = state.calc_col(cur_delta[id] + pos);
                        res_moves.push(GlobeMove::Rotate(real_pos));
                    }
                    iters[id] += 1;
                    found = true;
                    // eprintln!("Just skip: {id}");
                    break;
                }
            }
            if !found {
                break;
            }
        }
    }
    eprintln!("Total joined: {total_joined}");
    eprintln!("Total moves: {}", res_moves.len());
    res_moves
}

fn apply_row_moves(
    moves: &mut Vec<GlobeMove>,
    up_delta: i32,
    down_delta: i32,
    row1: usize,
    row2: usize,
) {
    for _cnt in 0..up_delta.abs() {
        moves.push(GlobeMove::RowShift {
            row: row1,
            delta: up_delta.signum(),
        });
    }
    for _cnt in 0..down_delta.abs() {
        moves.push(GlobeMove::RowShift {
            row: row2,
            delta: down_delta.signum(),
        });
    }
}

pub fn globe_optimize(data: &Data, task_types: &[&str], log: &mut SolutionsLog) {
    let mut solutions = TaskSolution::all_by_types(data, task_types);
    eprintln!("Number of tasks: {}", solutions.len());
    // solutions.truncate(1);

    for sol in solutions.iter_mut() {
        let mut rng = StdRng::seed_from_u64(785834334);
        eprintln!(
            "Solving task {}. Type: {}, {}",
            sol.task_id,
            sol.task.puzzle_type,
            sol.task.get_color_type()
        );

        let base_solution = data.solutions.my_138k[&sol.task_id].clone();
        eprintln!("Base solution len: {}", base_solution.len());

        let puzzle_type = &data.puzzle_info[&sol.task.puzzle_type];
        let mut state = GlobeState::new(puzzle_type);

        let layers = parse_existing_solution(&base_solution, &state);
        let layers = match layers {
            Some(layers) => layers,
            None => {
                eprintln!("Failed to parse existing solution!");
                continue;
            }
        };
        let combined_moves = combine_layers(&layers, &state);
        for mv in combined_moves.iter() {
            match *mv {
                GlobeMove::Rotate(col) => state.move_rotate(sol, col as usize),
                GlobeMove::RowShift { row, delta } => {
                    if delta == 1 {
                        state.move_row_right(sol, row, 1);
                    } else {
                        assert_eq!(delta, -1);
                        state.move_row_left(sol, row, 1);
                    }
                }
            }
        }

        globe_final_rows_move(&state, sol);
        state.show_state(&sol.state, &sol.target_state);
        assert!(sol.is_solved());
        eprintln!("Sol len [task={}]: {}", sol.task_id, sol.answer.len());
    }
    for sol in solutions.iter() {
        if sol.is_solved_with_wildcards() {
            log.append(sol);
        } else {
            eprintln!("Failed to solve task {}?!", sol.task_id);
        }
    }
}
