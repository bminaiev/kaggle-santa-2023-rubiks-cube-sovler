use rand::Rng;

use crate::{moves::SeveralMoves, permutation::Permutation, puzzle::Puzzle};

pub fn get_blocks(n: usize, moves: &[Permutation]) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();

    let mut blocks = vec![];
    let mut magic = vec![0; n];
    for mov in moves.iter() {
        let xor: u64 = rng.gen();
        for cycle in mov.cycles.iter() {
            for &x in cycle.iter() {
                magic[x] ^= xor;
            }
        }
    }
    let mut seen = vec![false; n];
    for i in 0..magic.len() {
        if seen[i] {
            continue;
        }
        let mut group = vec![];
        for j in 0..magic.len() {
            if magic[j] == magic[i] {
                group.push(j);
                seen[j] = true;
            }
        }
        blocks.push(group);
    }
    blocks
}

pub fn get_blocks_by_several_moves(n: usize, moves: &[SeveralMoves]) -> Vec<Vec<usize>> {
    get_blocks(
        n,
        &moves
            .iter()
            .map(|sv| sv.permutation.clone())
            .collect::<Vec<_>>(),
    )
}

pub fn get_start_permutation(task: &Puzzle, solution: &[String]) -> Vec<usize> {
    let n = task.info.n;
    let mut state: Vec<_> = (0..n).collect();
    for step in solution.iter() {
        let perm = &task.info.moves[step];
        for cycle in perm.cycles.iter() {
            for w in cycle.windows(2) {
                state.swap(w[0], w[1]);
            }
        }
    }
    let mut inv = vec![0; n];
    for (i, &x) in state.iter().enumerate() {
        inv[x] = i;
    }
    inv
}

pub fn get_all_perms(a: &[usize]) -> Vec<Vec<usize>> {
    let mut res = vec![];
    if a.len() == 1 {
        res.push(a.to_vec());
    } else if a.len() == 2 {
        res.push(a.to_vec());
        res.push(vec![a[1], a[0]]);
    } else if a.len() == 3 {
        res.push(a.to_vec());
        res.push(vec![a[1], a[2], a[0]]);
        res.push(vec![a[2], a[0], a[1]]);
        res.push(vec![a[2], a[1], a[0]]);
        res.push(vec![a[0], a[2], a[1]]);
        res.push(vec![a[1], a[0], a[2]]);
    } else {
        panic!();
    }
    res
}

pub fn perm_inv(a: &[usize]) -> Vec<usize> {
    let mut res = vec![0; a.len()];
    for (i, &x) in a.iter().enumerate() {
        res[x] = i;
    }
    res
}

pub fn perm_parity(perm: &[usize]) -> usize {
    let mut res = 0;
    for i in 0..perm.len() {
        for j in i + 1..perm.len() {
            if perm[i] > perm[j] {
                res += 1;
            }
        }
    }
    res % 2
}

pub fn calc_cube_side_size(n: usize) -> usize {
    let mut sz = 1;
    while 6 * sz * sz < n {
        sz += 1;
    }
    sz
}

pub fn get_cube_side_moves(sz: usize) -> Vec<String> {
    let mut res = vec![];
    for sign in ["", "-"] {
        for mv in ["d", "f", "r"] {
            for x in [0, sz - 1].iter() {
                let name = format!("{sign}{mv}{x}");
                res.push(name);
            }
        }
    }
    res
}
