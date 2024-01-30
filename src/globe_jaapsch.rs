use std::time::Instant;

use rand::{
    rngs::StdRng,
    seq::{IteratorRandom, SliceRandom},
    Rng, SeedableRng,
};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    data::Data,
    puzzle_type::PuzzleType,
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
    utils::{calc_num_invs, pick_random_perm},
};

pub struct GlobeState {
    pub n_rows: usize,
    pub n_cols: usize,
    pub puzzle_type: PuzzleType,
}

impl GlobeState {
    pub fn new(puzzle_type: &PuzzleType) -> Self {
        let mut n_rows = 0;
        let mut n_cols = 0;

        for (k, _v) in puzzle_type.moves.iter() {
            if k.starts_with('r') {
                n_rows += 1;
            } else if k.starts_with('f') {
                n_cols += 1;
            }
        }
        eprintln!("Size: {n_rows}x{n_cols}");
        Self {
            n_rows,
            n_cols,
            puzzle_type: puzzle_type.clone(),
        }
    }

    pub fn rc_to_index(&self, r: usize, c: usize) -> usize {
        r * self.n_cols + (c % self.n_cols)
    }

    pub fn index_to_rc(&self, i: usize) -> (usize, usize) {
        (i / self.n_cols, i % self.n_cols)
    }

    pub fn show_state(&self, a: &[usize], target_state: &[usize]) {
        eprintln!("--------");
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let idx = self.rc_to_index(r, c);
                let correct = a[idx] == target_state[idx];
                eprint!(
                    "{}{:2}{} ",
                    if correct { "[" } else { " " },
                    a[idx],
                    if correct { "]" } else { " " }
                );
            }
            eprintln!();
        }
        eprintln!("--------");
    }

    fn show_state_info(&self, a: &[&str]) {
        eprintln!("--------");
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                eprint!(" {:2} ", a[r * self.n_cols + c],);
            }
            eprintln!();
        }
        eprintln!("--------");
    }

    fn n(&self) -> usize {
        self.n_rows * self.n_cols
    }

    fn col_dist(&self, c1: usize, c2: usize) -> usize {
        let d = (c1 + self.n_cols - c2) % self.n_cols;
        d.min(self.n_cols - d)
    }

    fn col_dist_dir(&self, c1: usize, c2: usize) -> (usize, Dir) {
        let d = (c2 + self.n_cols - c1) % self.n_cols;
        if d == 0 {
            return (0, Dir::Stay);
        }
        if d < self.n_cols - d {
            (d, Dir::Right)
        } else {
            (self.n_cols - d, Dir::Left)
        }
    }

    pub fn move_row_right(&self, sol: &mut TaskSolution, row: usize, dist: usize) {
        assert!(dist <= self.n_cols);
        if dist < self.n_cols - dist {
            let mv = format!("-r{row}");
            for _ in 0..dist {
                sol.append_move(&mv);
            }
        } else {
            let mv = format!("r{row}");
            for _ in 0..(self.n_cols - dist) {
                sol.append_move(&mv);
            }
        }
    }

    pub fn move_row_left(&self, sol: &mut TaskSolution, row: usize, dist: usize) {
        self.move_row_right(sol, row, self.n_cols - dist);
    }

    fn move_row_to_pos(&self, sol: &mut TaskSolution, row: usize, from: usize, to: usize) {
        let from = from % self.n_cols;
        let to = to % self.n_cols;
        let (d, dir) = self.col_dist_dir(from, to);
        if dir == Dir::Right || dir == Dir::Stay {
            self.move_row_right(sol, row, d);
        } else {
            self.move_row_right(sol, row, self.n_cols - d);
        }
    }

    pub fn move_rotate(&self, sol: &mut TaskSolution, col: usize) {
        let col = col % self.n_cols;
        let mv = format!("f{col}");
        sol.append_move(&mv);
    }

    pub(crate) fn calc_col(&self, offset: i32) -> i32 {
        let sz = self.n_cols as i32;
        ((offset % sz) + sz) % sz
    }

    pub fn apply_matching_res(&self, sol: &mut TaskSolution, res: &MatchingResult, r1: usize) {
        let r2 = self.n_rows - r1 - 1;
        for mv in res.moves.iter() {
            match mv {
                &MyGlobeMove::Rotate(c) => {
                    self.move_rotate(sol, c + 1);
                    self.move_row_right(sol, r1, 1);
                    self.move_rotate(sol, c + 1);
                }
                MyGlobeMove::BottomRowRight => {
                    self.move_row_right(sol, r2, 1);
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Swap {
    cost: usize,
    pos1: usize,
    pos2: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Dir {
    Left,
    Stay,
    Right,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Split {
    invs: usize,
    pos: usize,
}

impl Split {
    const INF: Self = Self {
        invs: usize::MAX / 2,
        pos: usize::MAX,
    };
}

fn get_best_splits(state: &GlobeState, a: &[usize], allow_bad_parity: bool) -> Vec<Split> {
    let mut by_row = vec![[Split::INF; 2]; state.n_rows];
    for row in 0..state.n_rows {
        for split in 0..state.n_cols {
            let perm: Vec<_> = (0..state.n_cols)
                .map(|c| a[state.rc_to_index(row, (c + split) % state.n_cols)])
                .collect();
            let invs = calc_num_invs(&perm);
            let split = Split { invs, pos: split };
            if by_row[row][invs % 2] > split {
                by_row[row][invs % 2] = split;
            }
        }
    }
    let mut res = vec![Split::INF; state.n_rows];
    for r1 in 0..state.n_rows / 2 {
        let r2 = state.n_rows - r1 - 1;
        let invs0 = by_row[r1][0].invs + by_row[r2][0].invs;
        let invs1 = by_row[r1][1].invs + by_row[r2][1].invs;
        if invs0 < invs1 {
            res[r1] = by_row[r1][0];
            res[r2] = by_row[r2][0];
        } else {
            res[r1] = by_row[r1][1];
            res[r2] = by_row[r2][1];
        }
        if allow_bad_parity {
            let invs2 = by_row[r1][0].invs + by_row[r2][1].invs;
            let invs3 = by_row[r1][1].invs + by_row[r2][0].invs;
            if invs2 < invs0 && invs2 < invs1 && invs2 < invs3 {
                res[r1] = by_row[r1][0];
                res[r2] = by_row[r2][1];
            } else if invs3 < invs0 && invs3 < invs1 {
                res[r1] = by_row[r1][1];
                res[r2] = by_row[r2][0];
            }
        }
        assert!(res[r1] != Split::INF);
        assert!(res[r2] != Split::INF);
    }
    res
}

fn make_correct_perm(state: &GlobeState, sol: &mut TaskSolution) -> bool {
    // state.ensure_correct_rows(sol);
    let best_splits = get_best_splits(state, &sol.state, !sol.exact_perm);
    if sol.exact_perm {
        for r in 0..state.n_rows / 2 {
            let invs1 = best_splits[r].invs;
            let invs2 = best_splits[state.n_rows - r - 1].invs;
            if invs1 % 2 != invs2 % 2 {
                unreachable!();
            }
        }
    }
    loop {
        let best_splits = get_best_splits(state, &sol.state, !sol.exact_perm);
        let should_swap = |r: usize, c: usize| -> bool {
            let split = best_splits[r];
            if (c + 1) % state.n_cols == split.pos {
                return false;
            }
            let v1 = sol.state[state.rc_to_index(r, c)];
            let v2 = sol.state[state.rc_to_index(r, c + 1)];
            v1 > v2
        };
        let ok_swap = |r: usize, c: usize| -> bool {
            let split = best_splits[r];
            if (c + 1) % state.n_cols == split.pos {
                return false;
            }
            let v1 = sol.state[state.rc_to_index(r, c)];
            let v2 = sol.state[state.rc_to_index(r, c + 1)];
            sol.exact_perm || v1 == v2
        };

        // let total_num_invs = best_splits.iter().map(|s| s.invs).sum::<usize>();
        // let invs: Vec<_> = best_splits.iter().map(|s| s.invs).collect();
        // eprintln!("Total num invs: {}. {invs:?}", total_num_invs);
        if best_splits.iter().all(|s| s.invs == 0) {
            break;
        }
        let mut swaps = vec![];
        for r1 in 0..state.n_rows / 2 {
            let r2 = state.n_rows - r1 - 1;
            let (inv1, inv2) = (best_splits[r1].invs, best_splits[r2].invs);
            if inv1 == 0 && inv2 == 0 {
                continue;
            }
            for c1 in 0..state.n_cols {
                // let id1 = sol.state[state.rc_to_index(r1, c1)];
                // assert_eq!(state.index_to_rc(id1).0, r1);
                let sw1 = should_swap(r1, c1);
                for c2 in 0..state.n_cols {
                    let sw2 = should_swap(r2, c2);
                    if !sw1 && !sw2 {
                        continue;
                    }
                    let mut good = sw1 && sw2;
                    if inv1 == 0 && ok_swap(r1, c1) {
                        good = true;
                    }
                    if inv2 == 0 && ok_swap(r2, c2) {
                        good = true;
                    }
                    if good {
                        swaps.push(Swap {
                            cost: state.col_dist(c1, c2),
                            pos1: state.rc_to_index(r1, c1),
                            pos2: state.rc_to_index(r2, c2),
                        });
                    }
                }
            }
        }
        swaps.sort();
        if swaps.is_empty() {
            break;
        }
        let swap = swaps[0];
        // eprintln!(
        //     "Want to swap: {} and {}. Cost: {}",
        //     sol.state[swap.pos1], sol.state[swap.pos2], swap.cost
        // );
        let (r1, c1) = state.index_to_rc(swap.pos1);
        let (r2, c2) = state.index_to_rc(swap.pos2);
        state.move_row_to_pos(sol, r2, c2, c1);
        state.move_rotate(sol, c1 + 1);
        state.move_row_right(sol, r1, 1);
        state.move_rotate(sol, c1 + 1);
        state.move_row_left(sol, r1, 1);
        state.move_row_right(sol, r2, 1);
        state.move_rotate(sol, c1 + 1);
        state.move_row_left(sol, r2, 1);
        state.move_rotate(sol, c1 + 1);
    }
    // eprintln!("Almost solved?");
    // state.show_state(&sol.state, &sol.target_state);
    true
}

pub fn globe_final_rows_move(state: &GlobeState, sol: &mut TaskSolution) {
    eprintln!("Last moves!");
    for row in 0..state.n_rows {
        let mut best = (usize::MAX, usize::MAX);
        for c in 0..state.n_cols {
            let cur_value = sol.state[state.rc_to_index(row, c)];
            let prev_value =
                sol.state[state.rc_to_index(row, (c + state.n_cols - 1) % state.n_cols)];
            if cur_value != prev_value {
                let now = (cur_value, c);
                if now < best {
                    best = now;
                }
            }
        }
        state.move_row_to_pos(sol, row, best.1, 0);
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MyGlobeMove {
    Rotate(usize),
    BottomRowRight,
}

fn solve_two_rows(state: &GlobeState, sol: &mut TaskSolution, r1: usize, rng: &mut StdRng) {
    let r2 = state.n_rows - r1 - 1;
    eprintln!("Start state: (r1 = {r1}, r2 = {r2})");

    let mut row_by_color = vec![usize::MAX; sol.task.num_colors];
    for row_id in 0..2 {
        let mut prev_color = 0;
        for col in 0..state.n_cols {
            let target_color = sol.target_state[state.rc_to_index([r1, r2][row_id], col)];
            assert!(
                row_by_color[target_color] == usize::MAX || row_by_color[target_color] == row_id
            );
            row_by_color[target_color] = row_id;
            assert!(target_color >= prev_color);
            prev_color = target_color;
        }
    }

    // eprintln!("Row by color: {row_by_color:?}");

    // state.show_state(&sol.state, &sol.target_state);
    let mut init_colors = vec![usize::MAX; state.n_cols * 2];
    for row_id in 0..2 {
        for c in 0..state.n_cols {
            let row = [r1, r2][row_id];
            init_colors[row_id * state.n_cols + c] = sol.state[state.rc_to_index(row, c)];
        }
    }

    // eprintln!("Init colors: {init_colors:?}");

    let res = find_best(rng, &init_colors, &row_by_color);

    state.apply_matching_res(sol, &res, r1);
    // eprintln!("Real state:");
    // state.show_state(&sol.state, &sol.target_state);

    // eprintln!("Min invs: {}", res.tot_invs);
}

#[derive(Clone, Debug)]
pub struct GlobeSaStage {
    pub in_row: Vec<Vec<usize>>,
}

#[derive(Clone)]
pub struct GlobeSolutionInfo {
    pub stages: Vec<GlobeSaStage>,
    pub perms: Vec<Vec<usize>>,
    pub init_colors: Vec<usize>,
}

impl GlobeSolutionInfo {
    pub fn eval(&self) -> MatchingResult {
        let sz = self.init_colors.len() / 2;
        let mut a0: Vec<_> = (0..sz).collect();
        let mut a1: Vec<_> = (sz..2 * sz).collect();

        let mut all_moves = vec![];

        for (stage, perm) in self.stages.iter().zip(self.perms.iter()) {
            let mut pairs = vec![usize::MAX; sz * 2];
            for i in 0..perm.len() {
                let x = stage.in_row[0][i];
                let y = stage.in_row[1][perm[i]];
                pairs[x] = y;
            }
            let stage_moves = apply_matching(&mut a0, &mut a1, pairs);
            all_moves.extend(stage_moves);
        }

        let mut real_colors = vec![];
        for row_id in 0..2 {
            for col in 0..sz {
                let v = [&a0, &a1][row_id][col];
                let color = self.init_colors[v];
                real_colors.push(color);
                // let correct_row = self.row_by_color[color];
                // assert_eq!(correct_row, row_id);
            }
        }

        let tot_invs =
            calc_num_invs_cycle(&real_colors[..sz]).max(calc_num_invs_cycle(&real_colors[sz..]));

        MatchingResult {
            moves: all_moves,
            tot_invs,
        }
    }

    pub fn run_sa(&mut self, rng: &mut StdRng) {
        let mut prev_score = self.eval().tot_invs;
        let start_invs = prev_score;
        eprintln!("Start invs: {}", prev_score);

        let mut best = (prev_score, self.clone());
        // // TODO: change
        const MAX_SEC: f64 = 1.0 * 1.0;
        let temp_start = 10.0f64;
        let temp_end = 0.2f64;
        let start = Instant::now();
        let log_every = 1.0f64.max(MAX_SEC / 10.0);
        let mut last_reported = 0.0;
        let mut iters = 0;
        loop {
            let elapsed_s = start.elapsed().as_secs_f64();
            if elapsed_s > MAX_SEC {
                break;
            }
            iters += 1;
            if elapsed_s > last_reported + log_every {
                last_reported = elapsed_s;
                eprintln!(
                    "Elapsed: {:.2} sec. Iters/s: {}, Start score: {}. Score: {}. Best: {}",
                    elapsed_s,
                    iters as f64 / elapsed_s,
                    start_invs,
                    prev_score,
                    best.0
                );
            }
            let elapsed_frac = elapsed_s / MAX_SEC;
            let temp = temp_start * (temp_end / temp_start).powf(elapsed_frac);
            let stage_id = rng.gen_range(0..self.stages.len());
            let perm_size = self.perms[stage_id].len();
            let pos1 = rng.gen_range(0..perm_size);
            let pos2 = rng.gen_range(0..perm_size);
            self.perms[stage_id].swap(pos1, pos2);
            let new_score = self.eval().tot_invs;
            if new_score < best.0 {
                best = (new_score, self.clone());
            }
            if new_score < prev_score
                || fastrand::f64() < ((prev_score as f64 - new_score as f64) / temp).exp()
            {
                // Using a new state!
                prev_score = new_score;
            } else {
                // Rollback
                self.perms[stage_id].swap(pos1, pos2);
            }
        }
        eprintln!("After local opt: {start_invs} -> {prev_score}.",);
        *self = best.1;
    }

    pub(crate) fn new(
        stages: Vec<GlobeSaStage>,
        init_colors: Vec<usize>,
        rng: &mut StdRng,
    ) -> Self {
        let perms: Vec<_> = stages
            .iter()
            .map(|stage| pick_random_perm(stage.in_row[0].len(), rng))
            .collect();
        Self {
            stages,
            init_colors,
            perms,
        }
    }
}

fn find_best(rng: &mut StdRng, init_colors: &[usize], row_by_color: &[usize]) -> MatchingResult {
    let sz = init_colors.len() / 2;

    let mut stages = vec![];
    {
        let mut in_row0 = vec![vec![]; 2];
        let mut in_row1 = vec![vec![]; 2];
        for row_id in 0..2 {
            for col in 0..sz {
                let idx = row_id * sz + col;
                let correct_row = row_by_color[init_colors[idx]];
                if correct_row == row_id {
                    in_row0[row_id].push(idx);
                }
                in_row1[1 - correct_row].push(idx);
            }
        }
        stages.push(GlobeSaStage { in_row: in_row0 });
        stages.push(GlobeSaStage { in_row: in_row1 });
    }
    // {
    //     let mut in_row0 = vec![vec![]; 2];
    //     let mut in_row1 = vec![vec![]; 2];
    //     let mut in_row2 = vec![vec![]; 2];
    //     for row_id in 0..2 {
    //         for col in 0..sz {
    //             let idx = row_id * sz + col;
    //             let correct_row = row_by_color[init_colors[idx]];
    //             if correct_row != row_id {
    //                 in_row0[row_id].push(idx);
    //             }
    //             in_row1[correct_row].push(idx);
    //             in_row2[1 - correct_row].push(idx);
    //         }
    //     }
    //     stages.push(Stage { in_row: in_row0 });
    //     stages.push(Stage { in_row: in_row1 });
    //     stages.push(Stage { in_row: in_row2 });
    // }

    let perms: Vec<_> = stages
        .iter()
        .map(|stage| pick_random_perm(stage.in_row[0].len(), rng))
        .collect();

    let mut sol_info = GlobeSolutionInfo {
        stages,
        perms,
        // row_by_color: row_by_color.to_vec(),
        init_colors: init_colors.to_vec(),
    };
    sol_info.run_sa(rng);
    sol_info.eval()
}

pub struct MatchingResult {
    pub moves: Vec<MyGlobeMove>,
    pub tot_invs: usize,
}

fn apply_matching(a0: &mut [usize], a1: &mut [usize], pairs: Vec<usize>) -> Vec<MyGlobeMove> {
    let sz = a0.len();
    let mut moves = vec![];
    let cnt_move = sz / 2 - 1;
    let mut need_more: usize = pairs.iter().filter(|&&x| x != usize::MAX).count();
    while need_more > 0 {
        let mut changed = false;

        let mut do_stuff = |c1: usize, c2: usize| {
            let v1 = a0[c1];
            let v2 = a1[c2];
            if pairs[v1] == v2 {
                moves.push(MyGlobeMove::Rotate(c1));
                // eprintln!("Switch {v1} and {v2}");
                changed = true;
                a0[c1] = v2;
                a1[c2] = v1;
                move_cycle_subsegm_right(a0, c1, cnt_move);
                move_cycle_subsegm_left(a1, c2, cnt_move);
                // show_table(&a);
                need_more -= 1;
            }
        };

        for c1 in 0..sz {
            do_stuff(c1, (c1 + 1) % sz);
        }
        if !changed {
            a1.rotate_right(1);
            moves.push(MyGlobeMove::BottomRowRight);
        }
    }
    moves
}

fn calc_num_invs_cycle(a: &[usize]) -> usize {
    let mut split = vec![0; a.len() + 1];
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            if a[i] < a[j] {
                split[i + 1] += 1;
                split[j + 1] -= 1;
            }
            if a[i] > a[j] {
                split[0] += 1;
                split[i + 1] -= 1;
                split[j + 1] += 1;
            }
        }
    }
    for i in 0..split.len() - 1 {
        split[i + 1] += split[i];
    }
    split.iter().min().unwrap().to_owned()
}

fn move_cycle_subsegm_right<T>(a: &mut [T], to: usize, cnt: usize) {
    let segm_len = cnt + 1;
    if segm_len <= to + 1 {
        let from = to + 1 - segm_len;
        a[from..to + 1].rotate_right(1);
    } else {
        let more = segm_len - (to + 1);
        a[0..to + 1].rotate_right(1);
        let sz = a.len();
        a[sz - more..sz].rotate_right(1);
        a.swap(0, sz - more);
    }
}

fn move_cycle_subsegm_left<T>(a: &mut [T], to: usize, cnt: usize) {
    let segm_len = cnt + 1;
    let sz = a.len();
    if to + segm_len <= sz {
        a[to..to + segm_len].rotate_left(1);
    } else {
        let more = segm_len - (a.len() - to);
        a[to..sz].rotate_left(1);
        a[0..more].rotate_left(1);
        a.swap(more - 1, sz - 1);
    }
}

fn show_table(a: &[Vec<usize>]) {
    eprintln!("--------");
    for r in 0..a.len() {
        for c in 0..a[r].len() {
            eprint!("{:2} ", a[r][c]);
        }
        eprintln!();
    }
    eprintln!("--------");
}

// https://www.jaapsch.net/puzzles/master.htm
pub fn solve_globe_jaapsch(data: &Data, task_types: &[&str], log: &mut SolutionsLog) {
    let mut solutions = TaskSolution::all_by_types(data, task_types);
    eprintln!("Number of tasks: {}", solutions.len());
    // solutions.truncate(1);

    solutions.par_iter_mut().for_each(|sol| {
        let mut rng = StdRng::seed_from_u64(785834334);
        eprintln!(
            "Solving task {}. Type: {}, {}",
            sol.task_id,
            sol.task.puzzle_type,
            sol.task.get_color_type()
        );

        let puzzle_type = &data.puzzle_info[&sol.task.puzzle_type];
        let mut state = GlobeState::new(puzzle_type);

        let mut found = false;
        for it in 0..500 {
            // eprintln!("START ITER {it}");
            let sol_copy = sol.clone();
            // make_random_moves(&state, sol, &mut rng);
            // move_to_correct_rows(&state, sol);

            eprintln!("Sol len after correct rows: {}", sol.answer.len());
            for r in 0..state.n_rows / 2 {
                solve_two_rows(&state, sol, r, &mut rng);
            }

            // try_improve_num_perms(&state, sol);

            eprintln!("Sol len after improving perms: {}", sol.answer.len());
            if !make_correct_perm(&state, sol) {
                *sol = sol_copy;
                continue;
            }

            found = true;
            // state.show_state(&sol.state);
            break;
        }
        if !found {
            eprintln!("Failed to find solution for task {}", sol.task_id);
            return;
        }
        globe_final_rows_move(&state, sol);
        state.show_state(&sol.state, &sol.target_state);
        assert!(sol.is_solved());
        eprintln!("Sol len [task={}]: {}", sol.task_id, sol.answer.len());
    });
    for sol in solutions.iter() {
        if sol.is_solved_with_wildcards() {
            log.append(sol);
        } else {
            eprintln!("Failed to solve task {}?!", sol.task_id);
        }
    }
}

fn make_random_moves(state: &GlobeState, sol: &mut TaskSolution, rng: &mut StdRng) {
    for _ in 0..rng.gen_range(0..5) {
        let mv = state.puzzle_type.moves.keys().choose(rng).unwrap();
        sol.append_move(mv);
    }
}
