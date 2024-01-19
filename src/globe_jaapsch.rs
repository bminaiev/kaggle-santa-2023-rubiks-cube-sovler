use std::{
    cmp::{max, min},
    time::Instant,
};

use rand::{
    rngs::StdRng,
    seq::{IteratorRandom, SliceRandom},
    Rng, SeedableRng,
};
use rayon::{
    iter::{
        IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    vec,
};

use crate::{
    data::Data, puzzle_type::PuzzleType, sol_utils::TaskSolution, solutions_log::SolutionsLog,
    utils::calc_num_invs,
};

struct State {
    n_rows: usize,
    n_cols: usize,
    puzzle_type: PuzzleType,
}

impl State {
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

    fn rc_to_index(&self, r: usize, c: usize) -> usize {
        r * self.n_cols + (c % self.n_cols)
    }

    fn index_to_rc(&self, i: usize) -> (usize, usize) {
        (i / self.n_cols, i % self.n_cols)
    }

    fn show_state(&self, a: &[usize]) {
        eprintln!("--------");
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let idx = self.rc_to_index(r, c);
                let correct_row = self.index_to_rc(a[idx]).0 == r;
                eprint!(
                    "{}{:2}{} ",
                    if correct_row { "[" } else { " " },
                    a[idx],
                    if correct_row { "]" } else { " " }
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

    fn move_row_right(&self, sol: &mut TaskSolution, row: usize, dist: usize) {
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

    fn move_row_left(&self, sol: &mut TaskSolution, row: usize, dist: usize) {
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

    fn move_rotate(&self, sol: &mut TaskSolution, col: usize) {
        let col = col % self.n_cols;
        let mv = format!("f{col}");
        sol.append_move(&mv);
    }

    fn ensure_correct_rows(&self, sol: &TaskSolution) {
        for i in 0..self.n() {
            let (r1, _c1) = self.index_to_rc(i);
            let (r2, _c2) = self.index_to_rc(sol.state[i]);
            assert_eq!(r1, r2);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Swap {
    cost: usize,
    pos1: usize,
    pos2: usize,
}

fn apply_swap(state: &State, sol: &mut TaskSolution, swap: &Swap) {
    let (r1, c1) = state.index_to_rc(swap.pos1);
    let (r2, c2) = state.index_to_rc(swap.pos2);
    state.move_row_to_pos(sol, r2, c2, c1 + 1);
    state.move_rotate(sol, c1 + 1);
    state.move_row_right(sol, r1, 1);
    state.move_rotate(sol, c1 + 1);
}

fn move_to_correct_rows(state: &State, sol: &mut TaskSolution) {
    loop {
        let mut swaps = vec![];
        for r1 in 0..state.n_rows / 2 {
            let r2 = state.n_rows - r1 - 1;
            for c1 in 0..state.n_cols {
                let id1 = sol.state[state.rc_to_index(r1, c1)];
                let expect_row1 = state.index_to_rc(id1).0;
                if expect_row1 == r1 {
                    continue;
                }
                assert_eq!(expect_row1, r2);
                for c2 in 0..state.n_cols {
                    let id2 = sol.state[state.rc_to_index(r2, c2)];
                    let expect_row2 = state.index_to_rc(id2).0;
                    if expect_row2 == r2 {
                        continue;
                    }
                    assert_eq!(expect_row2, r1);
                    let d = state.col_dist(c1 + 1, c2);
                    let swap = Swap {
                        cost: d,
                        pos1: state.rc_to_index(r1, c1),
                        pos2: state.rc_to_index(r2, c2),
                    };
                    swaps.push(swap);
                }
            }
        }

        let cur_estime = get_perms_moves_estimate(&sol.state, state, true);
        swaps.par_iter_mut().for_each(|swap| {
            let mut nsol = sol.clone();
            apply_swap(state, &mut nsol, swap);
            let new_estimate = get_perms_moves_estimate(&nsol.state, state, true);
            assert!(new_estimate >= cur_estime);
            swap.cost += new_estimate - cur_estime;
        });
        swaps.sort();
        if swaps.is_empty() {
            break;
        }
        let swap = swaps[0];
        apply_swap(state, sol, &swap);
    }
    state.ensure_correct_rows(sol);
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

fn get_best_splits(state: &State, a: &[usize], allow_bad_parity: bool) -> Vec<Split> {
    let mut by_row = vec![[Split::INF; 2]; state.n_rows];
    for row in 0..state.n_rows {
        for split in 0..state.n_cols {
            let perm: Vec<_> = (0..state.n_cols)
                .map(|c| a[state.rc_to_index(row, (c + split) % state.n_cols)])
                .collect();
            let mut invs = 0;
            let mut seen = vec![false; state.n_cols];
            for &idx in perm.iter() {
                let (r, c) = state.index_to_rc(idx);
                if r != row {
                    continue;
                }
                for c2 in c + 1..state.n_cols {
                    if seen[c2] {
                        invs += 1;
                    }
                }
                seen[c] = true;
            }
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

fn make_correct_perm(state: &State, sol: &mut TaskSolution) -> bool {
    state.ensure_correct_rows(sol);
    let best_splits = get_best_splits(state, &sol.state, false);
    for r in 0..state.n_rows / 2 {
        let invs1 = best_splits[r].invs;
        let invs2 = best_splits[state.n_rows - r - 1].invs;
        if invs1 % 2 != invs2 % 2 {
            unreachable!();
        }
    }
    loop {
        let best_splits = get_best_splits(state, &sol.state, false);
        let should_swap = |r: usize, c: usize| -> bool {
            let split = best_splits[r];
            if (c + 1) % state.n_cols == split.pos {
                return false;
            }
            let v1 = sol.state[state.rc_to_index(r, c)];
            let v2 = sol.state[state.rc_to_index(r, c + 1)];
            v1 > v2
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
            if min(inv1, inv2) == 0 && max(inv1, inv2) == 1 {
                unreachable!();
            }
            for c1 in 0..state.n_cols {
                let id1 = sol.state[state.rc_to_index(r1, c1)];
                assert_eq!(state.index_to_rc(id1).0, r1);
                let sw1 = should_swap(r1, c1);
                if best_splits[r1].pos == (c1 + 1) % state.n_cols {
                    continue;
                }
                if !sw1 && inv1 != 0 {
                    continue;
                }
                for c2 in 0..state.n_cols {
                    if best_splits[r2].pos == (c2 + 1) % state.n_cols {
                        continue;
                    }
                    let sw2 = should_swap(r2, c2);
                    if sw2 || (inv2 == 0 && sw1) {
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
    eprintln!("Almost solved?");
    true
}

fn final_rows_move(state: &State, sol: &mut TaskSolution) {
    eprintln!("Last moves!");
    for row in 0..state.n_rows {
        let col = (0..state.n_cols)
            .find(|&c| {
                let idx = state.rc_to_index(row, c);
                let (_r, c) = state.index_to_rc(sol.state[idx]);
                c == 0
            })
            .unwrap();
        state.move_row_to_pos(sol, row, col, 0);
    }
}

fn get_perms_moves_estimate(a: &[usize], state: &State, allow_bad_parity: bool) -> usize {
    let best_splits = get_best_splits(state, a, allow_bad_parity);
    let mut res = 0;
    for r in 0..state.n_rows / 2 {
        let invs1 = best_splits[r].invs;
        let invs2 = best_splits[state.n_rows - r - 1].invs;
        res += invs1.max(invs2);
    }
    res * 9
}

fn apply(sz: usize, cc: &[usize]) -> bool {
    let mut a = vec![0; sz];
    for &c in cc.iter() {
        let mut tmp = vec![0; sz / 2];
        for i in 0..sz / 2 {
            tmp[i] = 1 ^ a[(c + i) % sz];
        }
        tmp.reverse();
        for i in 0..sz / 2 {
            a[(c + i) % sz] = tmp[i];
        }
    }
    a.iter().all(|&x| x == 0)
}

fn apply_perm(sz: usize, cc: &[usize]) -> Vec<usize> {
    let mut a: Vec<_> = (0..sz).collect();
    for &c in cc.iter() {
        let mut tmp = vec![0; sz / 2];
        for i in 0..sz / 2 {
            tmp[i] = a[(c + i) % sz];
        }
        tmp.reverse();
        for i in 0..sz / 2 {
            a[(c + i) % sz] = tmp[i];
        }
    }
    a
}

fn is_almost_id_perm(a: &[usize]) -> bool {
    for i in 0..a.len() {
        if (a[i] + 1) % a.len() != a[(i + 1) % a.len()] {
            return false;
        }
    }
    true
}

fn try_improve_num_perms(state: &State, sol: &mut TaskSolution) {
    let mut estimate = get_perms_moves_estimate(&sol.state, state, false);
    eprintln!("Estimate: {}", estimate);
    state.ensure_correct_rows(sol);

    let mut good_shifts = vec![];
    let sz = state.n_cols;
    {
        let c0 = 0;
        for c1 in 0..sz {
            if c0 == c1 {
                continue;
            }
            for c2 in 0..sz {
                if c1 == c2 {
                    continue;
                }
                for c3 in 0..sz {
                    if c2 == c3 {
                        continue;
                    }
                    for c4 in 0..sz {
                        let check = [0, c1, c2, c3, c4];
                        if apply(sz, &check) {
                            let perm = apply_perm(sz, &check);
                            if is_almost_id_perm(&perm) {
                                continue;
                            }
                            eprintln!("WOW! {check:?}. Perm: {perm:?}");
                            good_shifts.push(check);
                        }
                    }
                }
            }
        }
    }

    loop {
        let mut changed = false;
        for shift in good_shifts.iter() {
            let mut nsol = sol.clone();
            for &c in shift.iter() {
                state.move_rotate(&mut nsol, c);
            }

            state.ensure_correct_rows(&nsol);
            let new_estimate = get_perms_moves_estimate(&nsol.state, state, false);
            eprintln!("New Estimate: {}", new_estimate);
            if new_estimate + nsol.answer.len() < estimate + sol.answer.len() {
                eprintln!("Apply: {} -> {}", estimate, new_estimate);
                *sol = nsol;
                changed = true;
                estimate = new_estimate;
            }
        }
        if !changed {
            break;
        }
    }
}

fn test(state: &State, sol: &mut TaskSolution) {
    sol.state = (0..state.n()).collect();
    state.show_state(&sol.state);

    state.move_rotate(sol, 0);
    state.move_row_right(sol, 0, 1);
    state.move_rotate(sol, 0);

    state.show_state(&sol.state);
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum MyMove {
    Rotate(usize),
    BottomRowRight,
}

fn solve_two_rows123(
    state: &State,
    sol: &mut TaskSolution,
    r1: usize,
    rng: &mut StdRng,
) -> MatchingResult {
    let r2 = state.n_rows - r1 - 1;
    // [need to switch from the top?; need to switch from the bottom?]
    let mut need_switches = vec![[0, 0]; state.n()];
    let mut all_ids = vec![];
    for &row in [r1, r2].iter() {
        for col in 0..state.n_cols {
            let value = sol.state[state.rc_to_index(row, col)];
            all_ids.push(value);
            let (r, _c) = state.index_to_rc(value);
            if r == row {
                need_switches[value][0] = 1;
                need_switches[value][1] = 1;
            } else if row == r1 {
                need_switches[value][0] = 1;
            } else {
                need_switches[value][1] = 1;
            }
        }
    }
    let mut a = vec![vec![0; state.n_cols]; 2];
    for row_id in 0..2 {
        for c in 0..state.n_cols {
            let row = [r1, r2][row_id];
            a[row_id][c] = sol.state[state.rc_to_index(row, c)];
        }
    }
    // eprintln!("Created all pairs...");
    show_table(&a);
    let mut right_moves = 0;
    let mut moves = vec![];
    let cnt_move = state.n_cols / 2 - 1;
    loop {
        if right_moves > 10 * state.n_cols {
            eprintln!("Too many right moves!");
            unreachable!()
        }
        if need_switches.iter().all(|p| p.iter().all(|&x| x == 0)) {
            eprintln!("All done!");
            break;
        }
        let mut changed = false;
        for c1 in 0..a[0].len() {
            let c2 = (c1 + 1) % state.n_cols;
            let v1 = a[0][c1];
            let v2 = a[1][c2];
            if need_switches[v1][0] > 0 && need_switches[v2][1] > 0 {
                let a_copy = a.clone();

                a[0][c1] = v2;
                a[1][c2] = v1;
                move_cycle_subsegm_right(&mut a[0], c1, cnt_move);
                move_cycle_subsegm_left(&mut a[1], c2, cnt_move);
                need_switches[v1][0] -= 1;
                need_switches[v2][1] -= 1;
                if is_valid_rows(&a, &need_switches) {
                    eprintln!("Switch {v1} and {v2}");
                    changed = true;
                    moves.push(MyMove::Rotate(c1));
                    show_table2(&a, &need_switches);
                } else {
                    a = a_copy;
                    need_switches[v1][0] += 1;
                    need_switches[v2][1] += 1;
                }
            }
        }
        if !changed {
            eprintln!("Move second row right...");
            a[1].rotate_right(1);
            moves.push(MyMove::BottomRowRight);
            right_moves += 1;
        }
    }
    unreachable!()
}

fn pick_random_perm(n: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut a: Vec<_> = (0..n).collect();
    a.shuffle(rng);
    a
}

fn solve_two_rows(state: &State, sol: &mut TaskSolution, r1: usize, rng: &mut StdRng) {
    state.ensure_correct_rows(sol);
    let r2 = state.n_rows - r1 - 1;
    eprintln!("Start state: (r1 = {r1}, r2 = {r2})");
    state.show_state(&sol.state);
    let mut a = vec![vec![0; state.n_cols]; 2];
    for row_id in 0..2 {
        for c in 0..state.n_cols {
            let row = [r1, r2][row_id];
            a[row_id][c] = sol.state[state.rc_to_index(row, c)];
        }
    }

    let (min_invs, perms) = find_best(rng, a.clone(), state);
    let res = calc_perms_score(&a, &perms, state);

    for mv in res.moves.iter() {
        match mv {
            &MyMove::Rotate(c) => {
                state.move_rotate(sol, c + 1);
                state.move_row_right(sol, r1, 1);
                state.move_rotate(sol, c + 1);
            }
            MyMove::BottomRowRight => {
                state.move_row_right(sol, r2, 1);
            }
        }
    }

    eprintln!("Real state:");
    state.show_state(&sol.state);

    eprintln!("Min invs: {min_invs}");
}

fn convert_perm_to_matching(
    a: &[Vec<usize>],
    perm: &[usize],
    n: usize,
    rev: bool,
) -> Vec<Vec<usize>> {
    let mut pairs = vec![vec![0; n]; n];
    for i in 0..a[0].len() {
        let j = perm[i];
        // pairs[a[1][j]][a[0][i]] += 1;
        pairs[a[0][i]][a[1][j]] += 1;
    }
    if rev {
        for i in 0..pairs.len() {
            for j in i + 1..pairs.len() {
                let tmp = pairs[i][j];
                pairs[i][j] = pairs[j][i];
                pairs[j][i] = tmp;
            }
        }
    }
    pairs
}

fn find_best(rng: &mut StdRng, a: Vec<Vec<usize>>, state: &State) -> (usize, Vec<Vec<usize>>) {
    let sz = a[0].len();
    let mut ps: Vec<_> = (0..2).map(|_| pick_random_perm(sz, rng)).collect();

    let na2 = calc_perms_score(&a, &ps, state).a;
    let mut prev_score = calc_invs_score(&na2);
    let start_invs = prev_score;
    let mut best = (prev_score, ps.clone());

    // TODO: change
    const MAX_SEC: f64 = 20.0;
    let temp_start = 10.0f64;
    let temp_end = 0.2f64;
    let start = Instant::now();
    loop {
        let elapsed_s = start.elapsed().as_secs_f64();
        if elapsed_s > MAX_SEC {
            break;
        }
        let elapsed_frac = elapsed_s / MAX_SEC;
        let temp = temp_start * (temp_end / temp_start).powf(elapsed_frac);
        let id = rng.gen_range(0..ps.len());
        let pos1 = rng.gen_range(0..sz);
        let pos2 = rng.gen_range(0..sz);
        let mut ps2 = ps.clone();
        ps2[id].swap(pos1, pos2);
        let na3 = calc_perms_score(&a, &ps2, state);
        let new_score = calc_invs_score(&na3.a);
        if new_score < best.0 {
            best = (new_score, ps2.clone());
        }
        if new_score < prev_score
            || fastrand::f64() < ((prev_score as f64 - new_score as f64) / temp).exp()
        {
            // Using a new state!
            prev_score = new_score;
            ps = ps2;
        } else {
            // Rollback
        }
    }
    eprintln!(
        "After local opt: {start_invs} -> {prev_score}. Best: {}",
        best.0
    );

    best
}

fn calc_perms_score(a: &[Vec<usize>], perms: &[Vec<usize>], state: &State) -> MatchingResult {
    let pairs0 = convert_perm_to_matching(a, &perms[0], state.n(), false);
    let pairs1 = convert_perm_to_matching(a, &perms[1], state.n(), true);

    let mut res = apply_matching(a.to_vec(), pairs0, state).unwrap();
    let mut res2 = apply_matching(res.a, pairs1, state).unwrap();

    res.moves.extend(res2.moves);
    res2.moves = res.moves;
    res2
}

fn calc_invs_score(a: &[Vec<usize>]) -> usize {
    calc_num_invs_cycle(&a[0]).max(calc_num_invs_cycle(&a[1]))
}

fn apply_p1_p2(mut a: Vec<Vec<usize>>, ps: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let sz = a[0].len();
    for p in ps.iter() {
        let mut na = vec![vec![0; sz]; 2];
        for i in 0..sz {
            let j = p[i];
            na[0][j] = a[0][i];
            na[1][i] = a[1][j];
        }
        a = na;
    }
    a
}

fn is_valid_rows(a: &[Vec<usize>], need_switches: &[[usize; 2]]) -> bool {
    for row_id in 0..1 {
        let mut stay = vec![];
        for c in 0..a[0].len() {
            let v = a[row_id][c];
            if need_switches[v][row_id] > 0 {
                continue;
            }
            stay.push(v);
        }
        if calc_num_invs_cycle(&stay) != 0 {
            return false;
        }
    }
    true
}

struct MatchingResult {
    moves: Vec<MyMove>,
    tot_invs: usize,
    a: Vec<Vec<usize>>,
}

fn apply_matching(
    mut a: Vec<Vec<usize>>,
    mut pairs: Vec<Vec<usize>>,
    state: &State,
) -> Option<MatchingResult> {
    let mut right_moves = 0;
    let mut moves = vec![];
    let cnt_move = state.n_cols / 2 - 1;
    let mut need_more: usize = pairs.iter().map(|p| p.iter().sum::<usize>()).sum();
    loop {
        if right_moves > 10 * state.n_cols {
            // eprintln!("Too many right moves!");
            return None;
        }
        if need_more == 0 {
            // eprintln!("All done!");
            break;
        }
        let mut changed = false;
        for c1 in 0..a[0].len() {
            let c2 = (c1 + 1) % state.n_cols;
            let v1 = a[0][c1];
            let v2 = a[1][c2];
            if pairs[v1][v2] > 0 {
                moves.push(MyMove::Rotate(c1));
                // eprintln!("Switch {v1} and {v2}");
                changed = true;
                a[0][c1] = v2;
                a[1][c2] = v1;
                move_cycle_subsegm_right(&mut a[0], c1, cnt_move);
                move_cycle_subsegm_left(&mut a[1], c2, cnt_move);
                pairs[v1][v2] -= 1;
                // show_table(&a);
                need_more -= 1;
            }
        }
        if !changed {
            // eprintln!("Move second row right...");
            a[1].rotate_right(1);
            // show_table(&a);
            moves.push(MyMove::BottomRowRight);
            right_moves += 1;
        }
    }
    // eprintln!("After matchin I expect:");
    // show_table(&a);
    let inv1 = calc_num_invs(&a[0]);
    let inv2 = calc_num_invs(&a[1]);
    // eprintln!("Total invs: {} + {} = {}", inv1, inv2, inv1 + inv2);
    Some(MatchingResult {
        moves,
        tot_invs: inv1 + inv2,
        a,
    })
}

fn calc_num_invs_cycle(a: &[usize]) -> usize {
    let mut res = usize::MAX;
    let mut a = a.to_vec();
    for _ in 0..a.len() {
        a.rotate_left(1);
        res = res.min(calc_num_invs(&a));
    }
    res
}

fn move_cycle_subsegm_right(a: &mut [usize], mut to: usize, cnt: usize) {
    for _ in 0..cnt {
        let prev = (to + a.len() - 1) % a.len();
        a.swap(prev, to);
        to = prev;
    }
}

fn move_cycle_subsegm_left(a: &mut [usize], mut to: usize, cnt: usize) {
    for _ in 0..cnt {
        let next = (to + 1) % a.len();
        a.swap(next, to);
        to = next;
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

fn show_table2(a: &[Vec<usize>], more: &[[usize; 2]]) {
    eprintln!("--------");
    for r in 0..a.len() {
        for c in 0..a[r].len() {
            let v = a[r][c];
            let stay = more[v][r] == 0;
            eprint!(
                "{}{:2}{} ",
                if stay { "[" } else { " " },
                a[r][c],
                if stay { "]" } else { " " }
            );
        }
        eprintln!();
    }
    eprintln!("--------");
}

fn generate_all_perms(n: usize, cur: &mut Vec<usize>, res: &mut Vec<Vec<usize>>) {
    if cur.len() == n {
        res.push(cur.clone());
        return;
    }
    for i in 0..n {
        if cur.contains(&i) {
            continue;
        }
        cur.push(i);
        generate_all_perms(n, cur, res);
        cur.pop();
    }
}

// https://www.jaapsch.net/puzzles/master.htm
pub fn solve_globe_jaapsch(data: &Data, task_type: &str, log: &mut SolutionsLog) {
    let mut solutions = TaskSolution::all_by_type(data, task_type, false);
    eprintln!("Number of tasks: {}", solutions.len());
    // solutions.truncate(1);

    let mut rng = StdRng::seed_from_u64(7854334);

    for sol in solutions.iter_mut() {
        eprintln!(
            "Solving task {}. Type: {}, {}",
            sol.task_id,
            sol.task.puzzle_type,
            sol.task.get_color_type()
        );

        let puzzle_type = &data.puzzle_info[&sol.task.puzzle_type];
        let mut state = State::new(puzzle_type);

        let mut found = false;
        for it in 0..500 {
            // eprintln!("START ITER {it}");
            let sol_copy = sol.clone();
            make_random_moves(&state, sol, &mut rng);
            move_to_correct_rows(&state, sol);

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
            continue;
        }
        final_rows_move(&state, sol);
        // state.show_state(&sol.state);
        assert!(sol.is_solved());
        eprintln!("Sol len: {}", sol.answer.len());
        log.append(sol);
    }
}

fn make_random_moves(state: &State, sol: &mut TaskSolution, rng: &mut StdRng) {
    for _ in 0..rng.gen_range(0..5) {
        let mv = state.puzzle_type.moves.keys().choose(rng).unwrap();
        sol.append_move(mv);
    }
}