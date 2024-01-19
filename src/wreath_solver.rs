use std::collections::{HashMap, HashSet};

use crate::{
    data::Data, permutation::Permutation, sol_utils::TaskSolution, solutions_log::SolutionsLog,
};

///
///  AAAAA BBBBB
/// A     C     B
/// A    B A    B
/// A    B A    B
/// A    B A    B
/// A     C     B
///  AAAAA BBBBB
fn show_state(sol: &TaskSolution, colors: &[usize]) {
    let moves = &sol.task.info.moves;
    let left_cycle = &moves[&"l".to_string()].cycles[0];
    let right_cycle = &moves[&"r".to_string()].cycles[0];
    let n = left_cycle.len();
    let same_id = (1..n)
        .find(|x| left_cycle.contains(x) && right_cycle.contains(x))
        .unwrap();
    let mid_h = same_id;
    let rows = mid_h + 4;
    let mid_w = (n - mid_h * 2 - 4) / 2 + 1;
    let cols = mid_w * 2 + 3;
    let mut field = vec![vec![usize::MAX; cols]; rows];
    for cycle_id in 0..2 {
        for i in 0..n {
            let h;
            let mut w;
            if i == 0 {
                h = mid_h + 2;
                w = mid_w + 1;
            } else if i == same_id + cycle_id {
                h = 1;
                w = mid_w + 1;
            } else if i < same_id + cycle_id {
                h = mid_h + 2 - i;
                w = mid_w + 2;
            } else if i < same_id + mid_w + 1 + cycle_id {
                let offset = i - same_id - cycle_id;
                h = 0;
                w = mid_w + 1 - offset;
            } else if i < same_id + mid_w + mid_h + 3 + cycle_id {
                let offset = i - same_id - mid_w - 1 - cycle_id;
                h = offset + 1;
                w = 0;
            } else {
                let offset = i - same_id - mid_w - mid_h - 3 - cycle_id;
                h = mid_h + 3;
                w = offset + 1;
            }
            if cycle_id == 1 {
                w = cols - 1 - w;
            }
            let idx = if cycle_id == 0 {
                left_cycle[i]
            } else {
                right_cycle[(n - i) % n]
            };
            let color = colors[idx];
            field[h][w] = color;
        }
    }
    for h in 0..field.len() {
        for w in 0..field[h].len() {
            if field[h][w] == usize::MAX {
                print!(" ");
            } else {
                let c = (b'A' + field[h][w] as u8) as char;
                print!("{}", c);
            }
        }
        println!();
    }
}

pub fn solve_wreath(data: &Data, log: &mut SolutionsLog) {
    let mut tasks = TaskSolution::all_by_type(data, "wreath_100/100", false);

    let task = tasks[0].clone();

    eprintln!("Task {}. Type: {}", task.task_id, task.task.puzzle_type);
    for (k, v) in task.task.info.moves.iter() {
        eprintln!("{} -> {:?}", k, v);
    }

    let moves = &task.task.info.moves;

    let mut hm = HashSet::new();

    let mut check = |mvs: &[&str]| {
        let mut perm = Permutation::identity();
        for mv in mvs {
            let mv = moves.get(&mv.to_string()).unwrap();
            perm = perm.combine(&mv);
        }
        let sum_len = perm.sum_len();
        if sum_len < 10 {
            if hm.insert(perm.clone()) {
                eprintln!("{mvs:?} -> {:?}", perm.cycles);
            }
        }
    };

    // check(&[]);
    // for k1 in moves.keys() {
    //     check(&[k1]);
    //     for k2 in moves.keys() {
    //         check(&[k1, k2]);
    //         for k3 in moves.keys() {
    //             check(&[k1, k2, k3]);
    //             for k4 in moves.keys() {
    //                 check(&[k1, k2, k3, k4]);
    //                 for k5 in moves.keys() {
    //                     for k6 in moves.keys() {
    //                         // check(&[k1, k2, k3, k4, k5, k6]);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    let mut state: Vec<_> = task
        .state
        .iter()
        .map(|&p| task.task.solution_state[p])
        .collect();
    show_state(&task, &state);
}
