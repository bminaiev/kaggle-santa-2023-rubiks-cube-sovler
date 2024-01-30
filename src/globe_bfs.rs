use std::{collections::HashMap, mem, time::Instant};

use rand::{rngs::StdRng, SeedableRng};
use rustc_hash::FxHashMap;

use crate::{
    data::Data,
    globe_jaapsch::{globe_final_rows_move, GlobeState},
    globe_optimizer::GlobeMove,
    sol_utils::TaskSolution,
    solutions_log::SolutionsLog,
};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct State<const C: usize> {
    colors: [[u8; C]; 2],
}

impl<const C: usize> State<C> {
    fn set(&mut self, row: usize, col: usize, color: u8) {
        self.colors[row][col] = color;
    }

    pub fn new() -> Self {
        Self {
            colors: [[0; C]; 2],
        }
    }

    fn rotate(&mut self, pos: usize) {
        self.colors[0].rotate_left(pos);
        self.colors[1].rotate_left(pos);
        self.colors[0][..C / 2].reverse();
        self.colors[1][..C / 2].reverse();
        let (first, second) = self.colors.split_at_mut(1);
        first[0][..C / 2].swap_with_slice(&mut second[0][..C / 2]);
        self.colors[0].rotate_right(pos);
        self.colors[1].rotate_right(pos);
    }

    fn shift_right(&mut self, row: usize, delta: i32) {
        self.colors[row].rotate_right(delta as usize);
    }

    fn shift_left(&mut self, row: usize, delta: i32) {
        self.colors[row].rotate_left(delta as usize);
    }

    fn shift_row(&mut self, row: usize, delta: i32) {
        if delta > 0 {
            self.shift_right(row, delta);
        } else {
            self.shift_left(row, -delta);
        }
    }

    fn split_big(&self, big: &[u8], col: usize) -> Option<[u8; 4]> {
        let mut res = [u8::MAX; 4];
        for row in 0..2 {
            for i in 0..2 {
                let from = col + i * (C / 2);
                let mut big_here = u8::MAX;
                for j in from..from + C / 2 {
                    let value = self.colors[row][j % C];
                    if big.contains(&value) {
                        big_here = value;
                        break;
                    }
                }
                res[row * 2 + i] = big_here;
            }
        }
        if res.contains(&u8::MAX) {
            None
        } else {
            Some(res)
        }
    }

    fn print(&self) {
        for i in 0..2 {
            for j in 0..C {
                print!("{} ", self.colors[i][j],);
            }
            println!();
        }
    }

    fn num_splits(&self) -> usize {
        let mut res = 0;
        for row in 0..2 {
            for col in 0..C {
                let value1 = self.colors[row][col] as usize;
                let value2 = self.colors[row][(col + 1) % C] as usize;
                if value1 == 0 || value2 == 0 {
                    continue;
                }
                let mut bad = value1 / C != value2 / C;
                if (value1 + 1) % C != value2 % C {
                    bad = true;
                }
                if bad {
                    res += 1;
                }
            }
        }
        res
    }

    fn ok_rows(&self) -> bool {
        for row in 0..2 {
            for col in 0..C {
                let value = self.colors[row][col] as usize;
                if value / C != row {
                    return false;
                }
            }
        }
        true
    }
}

fn interesting_row(a: &[u8], can_remove: u8) -> bool {
    for pos in 0..a.len() {
        let mut tmp = a.to_vec();
        if can_remove != u8::MAX && tmp[pos] != can_remove {
            continue;
        }
        tmp.remove(pos);
        if tmp.windows(2).all(|a| a[0] < a[1]) {
            return true;
        }
    }
    false
}

struct PrevEdge<const C: usize> {
    dist: usize,
    prev_state: State<C>,
    rot_col: Option<usize>,
    row_shifts: [i32; 2],
}

fn bfs<const C: usize>(
    start_state: State<C>,
    max_levels: usize,
    rev_seen: &FxHashMap<State<C>, PrevEdge<C>>,
) -> (FxHashMap<State<C>, PrevEdge<C>>, Option<State<C>>) {
    let mut queues = vec![vec![]; max_levels + 1];
    let mut seen = FxHashMap::default();
    queues[0].push(start_state);
    seen.insert(
        start_state,
        PrevEdge {
            dist: 0,
            prev_state: start_state,
            rot_col: None,
            row_shifts: [0; 2],
        },
    );
    let start = Instant::now();
    let mut showed = 0;
    let mut seen_0_pos = vec![false; C];
    for lvl in 0..queues.len() {
        let mut cur_lvl = vec![];
        mem::swap(&mut queues[lvl], &mut cur_lvl);
        if !cur_lvl.is_empty() {
            eprintln!(
                "Level: {}. Size: {}. Elapsed: {:?}",
                lvl,
                cur_lvl.len(),
                start.elapsed()
            );
        }
        for state in cur_lvl {
            {
                if state.ok_rows()
                    && interesting_row(&state.colors[0], 0)
                    && interesting_row(&state.colors[1], u8::MAX)
                // && state.colors[0][4] == 0
                {
                    let index_of_0 = state.colors[0].iter().position(|&x| x == 0).unwrap();
                    if seen_0_pos[index_of_0] {
                        continue;
                    }
                    seen_0_pos[index_of_0] = true;
                    showed += 1;
                    eprintln!("Cur state: (splits = {})", state.num_splits());
                    state.print();
                }
            }
            if rev_seen.contains_key(&state) {
                eprintln!("Found solution!");
                return (seen, Some(state));
            }
            if lvl == queues.len() - 1 {
                continue;
            }
            if state.num_splits() > 4 {
                continue;
            }
            for row in 0..2 {
                for &delta in [-1, 1].iter() {
                    let mut new_state = state;
                    new_state.shift_row(row, delta);
                    let ncost = lvl + 1;
                    if ncost >= queues.len() {
                        continue;
                    }
                    if seen.contains_key(&new_state) && seen[&new_state].dist <= ncost {
                        continue;
                    }
                    let mut row_shifts = [0; 2];
                    row_shifts[row] = delta;
                    seen.insert(
                        new_state,
                        PrevEdge {
                            dist: ncost,
                            prev_state: state,
                            rot_col: None,
                            row_shifts,
                        },
                    );
                    queues[ncost].push(new_state);
                }
            }
            for col in 0..C {
                let range = -(C as i32) / 2 + 1..(C as i32) / 2;
                for d1 in range.clone() {
                    for d2 in range.clone() {
                        let mut new_state = state;
                        new_state.rotate(col);
                        new_state.shift_row(0, d1);
                        new_state.shift_row(1, d2);
                        new_state.rotate(col);
                        let cost =
                            lvl + 1 + 1 + d1.unsigned_abs() as usize + d2.unsigned_abs() as usize;
                        if cost >= queues.len() {
                            continue;
                        }
                        if seen.contains_key(&new_state) && seen[&new_state].dist <= cost {
                            continue;
                        }
                        seen.insert(
                            new_state,
                            PrevEdge {
                                dist: cost,
                                prev_state: state,
                                rot_col: Some(col),
                                row_shifts: [d1, d2],
                            },
                        );
                        queues[cost].push(new_state);
                    }
                }
            }
        }
    }
    eprintln!("Showed: {}", showed);
    (seen, None)
}

fn extract_path<const C: usize>(
    state: State<C>,
    seen: &FxHashMap<State<C>, PrevEdge<C>>,
    rev: bool,
) -> Vec<GlobeMove> {
    let mut res = vec![];
    let mut cur = state;
    while let Some(prev) = seen.get(&cur) {
        if prev.prev_state == cur {
            break;
        }
        if let Some(rot_col) = prev.rot_col {
            res.push(GlobeMove::Rotate(rot_col as i32));
        }
        for row in 0..2 {
            if prev.row_shifts[row] != 0 {
                res.push(GlobeMove::RowShift {
                    row,
                    delta: prev.row_shifts[row] * (if rev { -1 } else { 1 }),
                });
            }
        }
        if let Some(rot_col) = prev.rot_col {
            res.push(GlobeMove::Rotate(rot_col as i32));
        }
        cur = prev.prev_state;
    }
    if !rev {
        res.reverse();
    }
    res
}

#[derive(Default)]
pub struct Recolor {
    map: FxHashMap<usize, u8>,
}

impl Recolor {
    pub fn get_or_add(&mut self, color: usize) -> u8 {
        let sz = self.map.len();
        self.map.entry(color).or_insert_with(|| sz as u8);
        self.map[&color]
    }

    pub fn get(&self, color: usize) -> u8 {
        self.map[&color]
    }

    pub(crate) fn num_colors(&self) -> usize {
        self.map.len()
    }
}

fn solve_row<const C: usize>(
    sol: &mut TaskSolution,
    state: &GlobeState,
    row: usize,
    precalcs: &mut Precalcs<C>,
) -> Option<Vec<GlobeMove>> {
    assert_eq!(C, state.n_cols);
    let row2 = state.n_rows - row - 1;

    let mut recolor = Recolor::default();

    let mut final_state = State::<C>::new();
    for i in 0..2 {
        for j in 0..C {
            let color = recolor.get_or_add(sol.target_state[state.rc_to_index([row, row2][i], j)]);
            final_state.set(i, j, color);
        }
    }
    let mut start_state = State::<C>::new();
    for i in 0..2 {
        for j in 0..C {
            let color = recolor.get(sol.state[state.rc_to_index([row, row2][i], j)]);
            start_state.set(i, j, color);
        }
    }

    eprintln!("Solving this row:");
    start_state.print();

    eprintln!("Want to see this:");
    final_state.print();

    let from_final = precalcs.get_or_create(final_state);
    eprintln!("Precalc finished. Searching for solution...");
    let (from_start, mid_state) = bfs(start_state, LAYERS, from_final);
    if let Some(mid_state) = mid_state {
        eprintln!("Found solution! Wow!");
        let mut path = extract_path(mid_state, &from_start, false);
        path.extend(extract_path(mid_state, from_final, true));
        Some(path)
    } else {
        eprintln!("Failed to solve row!");
        None
    }
}

fn solve_c<const C: usize>(
    sol: &mut TaskSolution,
    state: &GlobeState,
    precalcs: &mut Precalcs<C>,
) -> bool {
    assert_eq!(C, state.n_cols);
    for row in 0..state.n_rows / 2 {
        if let Some(mut moves) = solve_row::<C>(sol, state, row, precalcs) {
            let row2 = state.n_rows - row - 1;
            for mv in moves.iter_mut() {
                mv.renumerate_rows(row, row2);
                for sub_move in mv.to_strings() {
                    sol.append_move(&sub_move);
                }
            }
            eprintln!("After solving row {row}:");
            state.show_state(&sol.state, &sol.target_state);
        } else {
            eprintln!("Failed to solve row {row}");
            return false;
        }
    }
    true
}

struct Precalcs<const C: usize> {
    by_state: FxHashMap<State<C>, FxHashMap<State<C>, PrevEdge<C>>>,
}

impl<const C: usize> Precalcs<C> {
    fn new() -> Self {
        Self {
            by_state: FxHashMap::default(),
        }
    }

    fn get_or_create(&mut self, state: State<C>) -> &FxHashMap<State<C>, PrevEdge<C>> {
        self.by_state.entry(state).or_insert_with(|| {
            eprintln!("No precalc for pos:");
            state.print();
            let (seen, _) = bfs(state, LAYERS, &FxHashMap::default());
            seen
        });
        self.by_state.get(&state).unwrap()
    }
}

const LAYERS: usize = 13;
fn test() -> bool {
    const C: usize = 16;
    let mut state = State::<C>::new();
    let mut start = [[0; C]; 2];
    for i in 0..2 {
        for j in 0..C {
            start[i][j] = i * C + j;
        }
    }
    for i in 0..2 {
        for j in 0..C {
            state.set(i, j, start[i][j].try_into().unwrap());
        }
    }
    let (from_start, _) = bfs(state, 50, &FxHashMap::default());
    eprintln!("Full size: {}", from_start.len());
    true
}

pub fn solve_globe_bfs(data: &Data, task_types: &[&str], log: &mut SolutionsLog) {
    if test() {
        return;
    }
    let mut solutions = TaskSolution::all_by_types(data, task_types);
    let mut solutions: Vec<_> = solutions
        .into_iter()
        .filter(|s| s.task.get_color_type() == "N1")
        .collect();
    eprintln!("Number of tasks: {}", solutions.len());
    // solutions.truncate(1);

    const C: usize = 8;
    let mut precalcs = Precalcs::<C>::new();

    for sol in solutions.iter_mut() {
        let mut rng = StdRng::seed_from_u64(785834334);
        eprintln!(
            "Solving task {}. Type: {}, {}",
            sol.task_id,
            sol.task.puzzle_type,
            sol.task.get_color_type()
        );

        let puzzle_type = &data.puzzle_info[&sol.task.puzzle_type];
        let mut state = GlobeState::new(puzzle_type);

        if !solve_c::<C>(sol, &state, &mut precalcs) {
            eprintln!("ooops failed to solve task {}!", sol.task_id);
            continue;
        }
        globe_final_rows_move(&state, sol);

        state.show_state(&sol.state, &sol.target_state);
        assert!(sol.is_solved());
        eprintln!("Sol len [task={}]: {}", sol.task_id, sol.answer.len());
        if sol.is_solved_with_wildcards() {
            log.append(sol);
        } else {
            eprintln!("Failed to solve task {}?!", sol.task_id);
        }
    }
}
