use crate::{sol_utils::TaskSolution, utils::perm_parity};

pub fn triangle_parity_solver(
    start_state: &[usize],
    groups: Vec<Vec<usize>>,
    sol: &TaskSolution,
    sz: usize,
    only_side_moves: bool,
) -> Vec<String> {
    let mut moves = vec![];
    for dir in ["f", "r", "d"].iter() {
        for x in 0..sz {
            if x * 2 + 1 == sz {
                continue;
            }
            if only_side_moves && (x != 0 && x != sz - 1) {
                continue;
            }
            moves.push(format!("{dir}{x}"));
        }
    }
    let calc_parity = |state: &[usize]| -> Vec<usize> {
        let mut res = vec![];
        for group in groups.iter() {
            let mut perm = vec![];
            for &pos in group.iter() {
                perm.push(state[pos]);
            }
            res.push(perm_parity(&perm));
        }
        res.extend(vec![0; moves.len()]);
        res
    };

    let mut matrix = vec![];
    for i in 0..moves.len() {
        let mut state: Vec<_> = (0..start_state.len()).collect();
        // eprintln!("move: {}", moves[i]);
        sol.task.info.moves[&moves[i]].apply(&mut state);
        let mut parity = calc_parity(&state);
        parity[groups.len() + i] = 1;
        matrix.push(parity);
    }
    matrix.push(calc_parity(start_state));
    let mut sz = 0;
    for col in 0..groups.len() {
        let mut row = sz;
        while row < matrix.len() - 1 && matrix[row][col] == 0 {
            row += 1;
        }
        if row == matrix.len() - 1 {
            continue;
        }
        matrix.swap(sz, row);
        for row in sz + 1..matrix.len() {
            if matrix[row][col] == 0 {
                continue;
            }
            for col in 0..matrix[row].len() {
                matrix[row][col] ^= matrix[sz][col];
            }
        }
        sz += 1;
    }
    for gr in 0..groups.len() {
        assert_eq!(matrix[matrix.len() - 1][gr], 0);
    }
    let mut res = vec![];
    for i in 0..moves.len() {
        if matrix[matrix.len() - 1][groups.len() + i] == 1 {
            res.push(moves[i].clone());
        }
    }
    res
}
