use std::{collections::HashMap, mem};

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    cube_edges_calculator::{build_squares, calc_cube_centers},
    permutation::Permutation,
    sol_utils::TaskSolution,
    utils::{calc_cube_side_size, show_cube_ids, slice_hash},
};

fn get_allowed_moves(
    sz: usize,
    rng: &mut StdRng,
    sol: &TaskSolution,
) -> Vec<(String, Permutation)> {
    let mut moves = vec![];
    for dir in ["f", "r", "d"].iter() {
        for x in 0..sz {
            if x * 2 + 1 == sz {
                continue;
            }
            moves.push(format!("{dir}{x}"));
            moves.push(format!("-{dir}{x}"));
        }
    }
    moves.shuffle(rng);
    moves
        .into_iter()
        .map(|name| (name.clone(), sol.task.info.moves[&name].clone()))
        .collect()
}

pub fn greedy_cube_optimizer_old(sol: &mut TaskSolution) {
    eprintln!("Start greedy");
    sol.show();
    let n = sol.state.len();
    let sz = calc_cube_side_size(n);
    let squares = build_squares(sz);
    let centers = calc_cube_centers(&squares);
    let target_state = sol.target_state.clone();
    let calc_score = |state: &[usize]| {
        let mut res = 0;
        for i in 0..state.len() {
            if centers[i] && state[i] != target_state[i] {
                res += 1;
            }
        }
        res
    };
    let show = |state: &[usize]| {
        let score = calc_score(state);
        eprintln!("Score: {score}");
        let ids: Vec<_> = (0..n)
            .filter(|&x| centers[x] && state[x] != target_state[x])
            .collect();
        show_cube_ids(&ids, sz);
    };
    let mut rng = StdRng::seed_from_u64(787788);
    let allowed_moves = get_allowed_moves(sz, &mut rng, sol);

    eprintln!("Start score: {}", calc_score(&sol.state));
    let mut queues = vec![vec![]; 500];
    queues[0].push(sol.state.clone());
    const MAX_CHECK: usize = 10434;
    let mut prev = HashMap::new();
    {
        let h = slice_hash(&sol.state);
        prev.insert(h, (h, usize::MAX));
    }
    let mut scores = vec![];
    for lvl in 0..queues.len() {
        let mut cur_lvl = vec![];
        mem::swap(&mut queues[lvl], &mut cur_lvl);
        cur_lvl.sort_by_cached_key(|a| calc_score(a));
        cur_lvl.truncate(MAX_CHECK);
        {
            let now_score = calc_score(&cur_lvl[0]);
            scores.push(now_score);
            eprintln!("Lvl: {lvl}. Score: {now_score}");
            // show(&cur_lvl[0]);
            if lvl == queues.len() - 1 || (lvl > 4 && scores[lvl - 4] <= now_score) {
                eprintln!("Finishing greedy...");
                let mut cur_hash = slice_hash(&cur_lvl[0]);
                let mut path = vec![];
                loop {
                    let (prev_hash, mv_id) = prev[&cur_hash];
                    if mv_id == usize::MAX {
                        break;
                    }
                    path.push(mv_id);
                    cur_hash = prev_hash;
                }
                path.reverse();
                for &mv_id in path.iter() {
                    let (name, _perm) = &allowed_moves[mv_id];
                    eprintln!("Move: {}", name);
                    sol.append_move(name);
                }
                sol.show();
                return;
            }
        }
        let cnt_chunks = 10;
        let mut chunks = vec![vec![]; cnt_chunks];
        for (i, state) in cur_lvl.into_iter().enumerate() {
            chunks[i % cnt_chunks].push(state);
        }
        let next_lvl: Vec<_> = chunks
            .into_par_iter()
            .flat_map(|chunk| {
                let mut next = vec![];
                for state in chunk.into_iter() {
                    let prev_hash = slice_hash(&state);
                    for (mv_id, (_, perm)) in allowed_moves.iter().enumerate() {
                        let mut new_state = state.clone();
                        perm.apply(&mut new_state);
                        next.push((new_state, prev_hash, mv_id));
                    }
                }
                next.sort_by_cached_key(|(state, _, _)| calc_score(state));
                next.truncate(MAX_CHECK / (cnt_chunks / 2));
                next
            })
            .collect();
        for (state, ph, mv_id) in next_lvl.iter() {
            let nh = slice_hash(state);
            if prev.contains_key(&nh) {
                continue;
            }
            prev.insert(nh, (*ph, *mv_id));
            queues[lvl + 1].push(state.clone());
        }
    }
}

pub fn greedy_cube_optimizer(sol: &mut TaskSolution) {
    eprintln!("Start greedy");
    sol.show();
    let n = sol.state.len();
    let sz = calc_cube_side_size(n);
    let squares = build_squares(sz);
    let centers = calc_cube_centers(&squares);
    let target_state = sol.target_state.clone();
    let calc_score = |state: &[usize]| {
        let mut res = 0;
        for i in 0..state.len() {
            if centers[i] && state[i] != target_state[i] {
                res += 1;
            }
        }
        res
    };
    let show = |state: &[usize]| {
        let score = calc_score(state);
        eprintln!("Score: {score}");
        let ids: Vec<_> = (0..n)
            .filter(|&x| centers[x] && state[x] != target_state[x])
            .collect();
        show_cube_ids(&ids, sz);
    };
    let mut rng = StdRng::seed_from_u64(787788);
    let allowed_moves = get_allowed_moves(sz, &mut rng, sol);

    eprintln!("Start score: {}", calc_score(&sol.state));
    let mut queues = vec![vec![]; 500];
    queues[0].push(sol.state.clone());
    const MAX_CHECK: usize = 1044;
    const DEEP: usize = 6;
    const TRIES: usize = 10000;
    let mut prev = HashMap::new();
    {
        let h = slice_hash(&sol.state);
        prev.insert(h, (h, usize::MAX));
    }
    let mut scores = vec![];
    for lvl in 0..queues.len() {
        let mut cur_lvl = vec![];
        mem::swap(&mut queues[lvl], &mut cur_lvl);
        cur_lvl.sort_by_cached_key(|a| calc_score(a));
        cur_lvl.truncate(MAX_CHECK);
        {
            let now_score = calc_score(&cur_lvl[0]);
            scores.push(now_score);
            eprintln!("Lvl: {lvl}. Score: {now_score}");
            // show(&cur_lvl[0]);
            if lvl == queues.len() - 1 || (lvl > 4 && scores[lvl - 4] <= now_score) {
                eprintln!("Finishing greedy...");
                let mut cur_hash = slice_hash(&cur_lvl[0]);
                let mut path = vec![];
                loop {
                    let (prev_hash, mv_id) = prev[&cur_hash];
                    if mv_id == usize::MAX {
                        break;
                    }
                    path.push(mv_id);
                    cur_hash = prev_hash;
                }
                path.reverse();
                for &mv_id in path.iter() {
                    let (name, _perm) = &allowed_moves[mv_id];
                    eprintln!("Move: {}", name);
                    sol.append_move(name);
                }
                sol.show();
                return;
            }
        }
        let cnt_chunks = 10;
        let mut chunks = vec![vec![]; cnt_chunks];
        for (i, state) in cur_lvl.into_iter().enumerate() {
            chunks[i % cnt_chunks].push(state);
        }
        let chunks: Vec<_> = chunks
            .into_iter()
            .map(|c| (c, StdRng::seed_from_u64(rng.gen())))
            .collect();
        let next_lvl: Vec<_> = chunks
            .into_par_iter()
            .map(|(chunk, mut rng)| {
                let mut next = vec![];
                let mut history = HashMap::new();
                for state in chunk.into_iter() {
                    for _ in 0..TRIES {
                        let mut state = state.clone();
                        let mut prev_hash = slice_hash(&state);
                        for _ in 0..DEEP {
                            let mv_id = rng.gen_range(0..allowed_moves.len());
                            let (_, perm) = &allowed_moves[mv_id];
                            perm.apply(&mut state);
                            let nh = slice_hash(&state);
                            if history.contains_key(&nh) {
                                break;
                            }
                            history.insert(nh, (prev_hash, mv_id));
                            prev_hash = nh;
                        }
                        next.push(state);
                    }
                }
                next.sort_by_cached_key(|state| calc_score(state));
                next.truncate(MAX_CHECK / (cnt_chunks / 2));
                (next, history)
            })
            .collect();
        for (next, history) in next_lvl.into_iter() {
            for state in next.into_iter() {
                let nh = slice_hash(&state);
                if prev.contains_key(&nh) {
                    continue;
                }
                let mut cur_hash = nh;
                queues[lvl + 1].push(state);
                loop {
                    if prev.contains_key(&cur_hash) {
                        break;
                    }
                    let (prev_hash, mv_id) = history[&cur_hash];
                    prev.insert(cur_hash, (prev_hash, mv_id));
                    cur_hash = prev_hash;
                }
            }
        }
    }
}
