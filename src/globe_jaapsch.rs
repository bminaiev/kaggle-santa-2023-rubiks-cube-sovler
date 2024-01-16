use std::cmp::{max, min};

use rand::{seq::IteratorRandom, Rng};

use crate::{
    data::Data, puzzle_type::PuzzleType, sol_utils::TaskSolution, solutions_log::SolutionsLog,
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
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Swap {
    cost: usize,
    pos1: usize,
    pos2: usize,
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
                    swaps.push(Swap {
                        cost: d,
                        pos1: state.rc_to_index(r1, c1),
                        pos2: state.rc_to_index(r2, c2),
                    });
                }
            }
        }
        swaps.sort();
        if swaps.is_empty() {
            break;
        }
        let swap = swaps[0];
        let (r1, c1) = state.index_to_rc(swap.pos1);
        let (r2, c2) = state.index_to_rc(swap.pos2);
        state.move_row_to_pos(sol, r2, c2, c1 + 1);
        state.move_rotate(sol, c1 + 1);
        state.move_row_right(sol, r1, 1);
        state.move_rotate(sol, c1 + 1);
    }
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

fn get_best_split(state: &State, sol: &TaskSolution, row: usize) -> Split {
    let mut best_split = Split {
        invs: usize::MAX,
        pos: usize::MAX,
    };
    for split in 0..state.n_cols {
        let perm: Vec<_> = (0..state.n_cols)
            .map(|c| sol.state[state.rc_to_index(row, (c + split) % state.n_cols)])
            .collect();
        let mut invs = 0;
        let mut seen = vec![false; state.n_cols];
        for &idx in perm.iter() {
            let (_r, c) = state.index_to_rc(idx);
            for c2 in c + 1..state.n_cols {
                if seen[c2] {
                    invs += 1;
                }
            }
            seen[c] = true;
        }

        let split = Split { invs, pos: split };
        if split < best_split {
            best_split = split;
        }
    }
    best_split
}

fn make_correct_perm(state: &State, sol: &mut TaskSolution) -> bool {
    loop {
        let best_splits: Vec<_> = (0..state.n_rows)
            .map(|r| get_best_split(state, sol, r))
            .collect();
        let should_swap = |r: usize, c: usize| -> bool {
            let split = best_splits[r];
            if (c + 1) % state.n_cols == split.pos {
                return false;
            }
            let v1 = sol.state[state.rc_to_index(r, c)];
            let v2 = sol.state[state.rc_to_index(r, c + 1)];
            v1 > v2
        };
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
                eprintln!("Got into bad state...");
                return false;
            }
            for c1 in 0..state.n_cols {
                let id1 = sol.state[state.rc_to_index(r1, c1)];
                assert_eq!(state.index_to_rc(id1).0, r1);
                let sw1 = should_swap(r1, c1);
                if !sw1 && inv1 != 0 {
                    continue;
                }
                for c2 in 0..state.n_cols {
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

// https://www.jaapsch.net/puzzles/master.htm
pub fn solve_globe_jaapsch(data: &Data, task_type: &str, log: &mut SolutionsLog) {
    let mut solutions = TaskSolution::all_by_type(data, task_type, false);
    eprintln!("Number of tasks: {}", solutions.len());
    // solutions.truncate(1);

    for sol in solutions.iter_mut() {
        eprintln!(
            "Solving task {}. Type: {}, {}",
            sol.task_id,
            sol.task.puzzle_type,
            sol.task.get_color_type()
        );

        let puzzle_type = &data.puzzle_info[&sol.task.puzzle_type];
        let mut state = State::new(puzzle_type);
        // state.show_state(&sol.state);

        let mut found = false;
        for it in 0..500 {
            // eprintln!("START ITER {it}");
            let sol_copy = sol.clone();
            make_random_moves(&state, sol);
            move_to_correct_rows(&state, sol);

            // state.show_state(&sol.state);
            // eprintln!("Sol len: {}", sol.answer.len());
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

fn make_random_moves(state: &State, sol: &mut TaskSolution) {
    let mut rng = rand::thread_rng();
    for _ in 0..rng.gen_range(0..5) {
        let mv = state.puzzle_type.moves.keys().choose(&mut rng).unwrap();
        sol.append_move(mv);
    }
}
