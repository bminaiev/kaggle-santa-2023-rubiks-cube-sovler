use crate::{data::Data, puzzle_type::PuzzleType, sol_utils::TaskSolution, utils::perm_inv};

#[derive(Clone, Copy, Debug)]
struct NeededMoves {
    cnt_moves: i32,
    stay: usize,
}

#[derive(Clone, Debug)]
struct PossibleMove {
    need_moves: NeededMoves,
    score_delta: i32,
    rows: Vec<usize>,
}

struct State {
    n_rows: usize,
    n_cols: usize,
    next_pos: Vec<usize>,
    prev_pos: Vec<usize>,
    target_next: Vec<usize>,
    puzzle_type: PuzzleType,
}

impl State {
    pub fn new(puzzle_type: &PuzzleType) -> Self {
        let mut n_rows = 0;
        let mut n_cols = 0;

        for (k, v) in puzzle_type.moves.iter() {
            if k.starts_with("r") {
                n_rows += 1;
            } else if k.starts_with("f") {
                n_cols += 1;
            }
        }
        eprintln!("Size: {n_rows}x{n_cols}");
        let n = n_cols * n_rows;
        let mut res = Self {
            n_rows,
            n_cols,
            next_pos: vec![0; n],
            prev_pos: vec![0; n],
            target_next: vec![0; n],
            puzzle_type: puzzle_type.clone(),
        };
        for i in 0..n {
            let (r, c) = res.index_to_rc(i);
            let nc = if r < n_rows / 2 {
                (c + 1) % n_cols
            } else {
                (c + n_cols - 1) % n_cols
            };
            res.next_pos[i] = res.rc_to_index(r, nc);
        }
        res.target_next = res.next_pos.clone();
        res.prev_pos = perm_inv(&res.next_pos);
        res
    }

    fn rc_to_index(&self, r: usize, c: usize) -> usize {
        r * self.n_cols + (c % self.n_cols)
    }

    fn index_to_rc(&self, i: usize) -> (usize, usize) {
        (i / self.n_cols, i % self.n_cols)
    }

    fn show_state(&self, a: &[usize]) {
        println!("Current score: {}/{}", self.calc_score(a), self.n());
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let idx = self.rc_to_index(r, c);
                let good = self.target_next[a[idx]] == a[self.next_pos[idx]];
                print!(
                    "{}{:2}{} ",
                    if good { '[' } else { ' ' },
                    a[idx],
                    if good { ']' } else { ' ' }
                );
            }
            println!();
        }
    }

    fn calc_score(&self, a: &[usize]) -> usize {
        let mut res = 0;
        for pos in 0..a.len() {
            let who = a[pos];
            let real_next = a[self.next_pos[pos]];
            let target_next = self.target_next[who];
            res += if real_next == target_next { 1 } else { 0 };
        }
        res
    }

    fn calc_score_distr(&self, a: &[usize]) -> Vec<usize> {
        let mut by_row = vec![0; self.n_rows];
        for pos in 0..a.len() {
            let who = a[pos];
            let real_next = a[self.next_pos[pos]];
            let target_next = self.target_next[who];
            if real_next == target_next {
                by_row[self.index_to_rc(who).0] += 1;
            }
        }
        by_row
    }

    fn n(&self) -> usize {
        self.n_rows * self.n_cols
    }

    fn col_dist(&self, c1: usize, c2: usize) -> usize {
        let d = (c1 + self.n_cols - c2) % self.n_cols;
        d.min(self.n_cols - d)
    }

    fn calc_needed_moves(&self, rows: &[usize]) -> NeededMoves {
        let mut res = NeededMoves {
            cnt_moves: i32::MAX,
            stay: 0,
        };

        for &stay in rows.iter() {
            // for the last move
            let mut cur = 1;
            for &go in rows.iter() {
                let d = self.col_dist(go, stay);
                cur += d;
            }
            let cur = cur as i32;
            if cur < res.cnt_moves {
                res = NeededMoves {
                    cnt_moves: cur,
                    stay,
                };
            }
        }
        res
    }

    fn apply_move(&self, a: &mut [usize], mv: &PossibleMove) {
        let mut moves = vec![];
        for row in 0..mv.rows.len() {
            let mv_right = (mv.need_moves.stay + self.n_cols - mv.rows[row]) % self.n_cols;
            if mv_right * 2 < self.n_cols {
                for _ in 0..mv_right {
                    moves.push(format!("-r{row}"));
                }
            } else {
                for _ in 0..(self.n_cols - mv_right) {
                    moves.push(format!("r{row}"));
                }
            }
        }
        moves.push(format!("f{}", mv.need_moves.stay));
        // eprintln!("Appled moves: {moves:?}");
        for mv in moves.into_iter() {
            self.puzzle_type.moves[&mv].apply(a);
        }
    }
}

fn dfs(
    state: &State,
    more_moves: i32,
    a: &mut [usize],
    answer: &mut Vec<String>,
    base_score: usize,
    it: usize,
    more_cycles: i32,
) -> bool {
    if more_moves == 0 {
        return false;
    }
    if it < state.n_rows {
        for dx in -more_moves..=more_moves {
            if it == 0 && dx != 0 {
                continue;
            }
            let mv_name = if dx < 0 {
                format!("-r{it}")
            } else {
                format!("r{it}")
            };
            for _ in 0..dx.abs() {
                state.puzzle_type.moves[&mv_name].apply(a);
                answer.push(mv_name.clone());
            }
            if dfs(
                state,
                more_moves - dx.abs(),
                a,
                answer,
                base_score,
                it + 1,
                more_cycles,
            ) {
                return true;
            }
            for _ in 0..dx.abs() {
                state.puzzle_type.moves[&mv_name].apply_rev(a);
                answer.pop();
            }
        }
    } else {
        for col in 0..state.n_cols {
            let mv_name = format!("f{}", col);
            state.puzzle_type.moves[&mv_name].apply(a);
            answer.push(mv_name.clone());
            let score = state.calc_score(a);
            if score > base_score {
                return true;
            }
            if more_cycles > 1 {
                if dfs(
                    state,
                    more_moves - 1,
                    a,
                    answer,
                    base_score,
                    0,
                    more_cycles - 1,
                ) {
                    return true;
                }
            }
            state.puzzle_type.moves[&mv_name].apply_rev(a);
            answer.pop();
        }
    }
    false
}

pub fn solve_globe(data: &Data, task_type: &str) {
    let mut solutions = TaskSolution::all_by_type(data, task_type);
    eprintln!("Number of tasks: {}", solutions.len());
    solutions.truncate(1);

    let puzzle_type = &data.puzzle_info[task_type];
    let state = State::new(puzzle_type);

    let mut sol = &mut solutions[0];
    let target_next = state.next_pos.clone();

    let calc_score_one_cell = |idx: usize, next: usize| -> usize {
        if target_next[idx] == next {
            1
        } else {
            0
        }
    };

    let n = state.n();
    let mut start_score = state.calc_score(&sol.state);
    eprintln!("Initial score: {start_score}/{n}");
    state.show_state(&sol.state);

    for move_it in 0..250 {
        eprintln!("START {move_it}");
        for more_cycles in 1..=3 {
            let mut found = false;
            for more_moves in 1..30 {
                eprintln!("Trying {more_moves}x{more_cycles} moves");
                if dfs(
                    &state,
                    more_moves,
                    &mut sol.state,
                    &mut sol.answer,
                    start_score,
                    0,
                    more_cycles,
                ) {
                    found = true;
                    eprintln!("FOUND in {more_moves} moves!");
                    break;
                }
            }
            if found {
                break;
            }
        }
        // state.show_state(&sol.state);
        start_score = state.calc_score(&sol.state);
        eprintln!("Sol len: {}. Score = {start_score}/{}", sol.answer.len(), n);
    }

    let cmp = |mv1: &PossibleMove, mv2: &PossibleMove| {
        if mv1.score_delta.signum() != mv2.score_delta.signum() {
            return mv1
                .score_delta
                .signum()
                .cmp(&mv2.score_delta.signum())
                .reverse();
        }
        (mv1.score_delta * mv2.need_moves.cnt_moves)
            .cmp(&(mv2.score_delta * mv1.need_moves.cnt_moves))
            .reverse()
    };

    // for it in 0..200 {
    //     let mut possible_moves = vec![];

    //     let a = &mut sol.state;
    //     const BIG: usize = 10;
    //     for r1 in 0..n_cols {
    //         for r2 in 0..n_cols {
    //             if col_dist(r1, r2) > BIG {
    //                 continue;
    //             }
    //             for r3 in 0..n_cols {
    //                 if col_dist(r2, r3) > BIG || col_dist(r1, r3) > BIG {
    //                     continue;
    //                 }
    //                 for r4 in 0..n_cols {
    //                     if col_dist(r3, r4) > BIG
    //                         || col_dist(r2, r4) > BIG
    //                         || col_dist(r1, r4) > BIG
    //                     {
    //                         continue;
    //                     }
    //                     let p1 = rc_to_index(0, r1);
    //                     let p2 = rc_to_index(1, r2);
    //                     let p3 = rc_to_index(2, r3);
    //                     let p4 = rc_to_index(3, r4);

    //                     let p1_end = rc_to_index(0, r1 + n_cols / 2);
    //                     let p2_end = rc_to_index(1, r2 + n_cols / 2);
    //                     let p3_end = rc_to_index(2, r3 + n_cols / 2);
    //                     let p4_end = rc_to_index(3, r4 + n_cols / 2);

    //                     let mut prev_score = 0;
    //                     prev_score += calc_score_one_cell(a[prev_pos[p1]], a[p1]);
    //                     prev_score += calc_score_one_cell(a[prev_pos[p2]], a[p2]);
    //                     prev_score += calc_score_one_cell(a[p3], a[next_pos[p3]]);
    //                     prev_score += calc_score_one_cell(a[p4], a[next_pos[p4]]);
    //                     prev_score += calc_score_one_cell(a[prev_pos[p1_end]], a[p1_end]);
    //                     prev_score += calc_score_one_cell(a[prev_pos[p2_end]], a[p2_end]);
    //                     prev_score += calc_score_one_cell(a[p3_end], a[next_pos[p3_end]]);
    //                     prev_score += calc_score_one_cell(a[p4_end], a[next_pos[p4_end]]);

    //                     let mut new_score = 0;
    //                     new_score += calc_score_one_cell(a[prev_pos[p1]], a[next_pos[p4_end]]);
    //                     new_score += calc_score_one_cell(a[prev_pos[p2]], a[next_pos[p3_end]]);
    //                     new_score += calc_score_one_cell(a[prev_pos[p2_end]], a[next_pos[p3]]);
    //                     new_score += calc_score_one_cell(a[prev_pos[p1_end]], a[next_pos[p4]]);
    //                     new_score += calc_score_one_cell(a[p4], a[p1_end]);
    //                     new_score += calc_score_one_cell(a[p3], a[p2_end]);
    //                     new_score += calc_score_one_cell(a[p3_end], a[p2]);
    //                     new_score += calc_score_one_cell(a[p4_end], a[p1]);

    //                     let score_delta = (new_score as i32) - (prev_score as i32);
    //                     let rows = vec![r1, r2, r3, r4];
    //                     if rows == vec![0, 0, 0, 0] {
    //                         continue;
    //                     }
    //                     let possible_move = PossibleMove {
    //                         need_moves: calc_needed_moves(&rows),
    //                         score_delta,
    //                         rows,
    //                     };
    //                     possible_moves.push(possible_move);
    //                 }
    //             }
    //         }
    //     }
    //     possible_moves.sort_by(cmp);

    //     let use_move = &possible_moves[0];

    //     eprintln!("{it}. Applying move: {use_move:?}");
    //     state.apply_move(a, use_move, &mut sol.answer);

    //     eprintln!(
    //         "Current score: {}/{n}. Ans len: {}. Score distr: {:?}",
    //         calc_score(a),
    //         sol.answer.len(),
    //         calc_score_distr(a)
    //     );
    // }

    eprintln!("Final:");
    state.show_state(&sol.state);

    for r in 0..state.n_rows {
        for c in 0..state.n_cols {
            let color =
                &sol.task.color_names[sol.task.solution_state[sol.state[state.rc_to_index(r, c)]]];
            print!("{color}");
        }
        println!()
    }

    // for mv in possible_moves[..10].iter() {
    //     eprintln!("{mv:?}");
    // }

    // for i in 0..possible_moves.len() {
    //     let use_move = &possible_moves[i];
    //     let mut tmp_state = sol.state.clone();
    //     apply_move(&mut tmp_state, use_move, &mut vec![]);

    //     let mut cur_score = calc_score(&tmp_state);
    //     let expected_score = ((start_score as i32) + use_move.score_delta) as usize;
    //     if cur_score != expected_score {
    //         eprintln!("{i}. Expected score: {expected_score}. Real score: {cur_score}");
    //         break;
    //     }
    //     // eprintln!("Score: {cur_score}/{n}");
    //     // show_state(&sol.state);
    // }

    eprintln!("Finished!")
}
