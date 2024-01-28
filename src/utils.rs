use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::{Range, RangeInclusive},
};

use rand::Rng;

use crate::{
    cube_edges_calculator::build_squares,
    moves::{rev_move, SeveralMoves},
    permutation::Permutation,
    puzzle::Puzzle,
    sol_utils::TaskSolution,
};

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

pub fn show_cube_ids(ids: &[usize], sz: usize) {
    let squares = build_squares(sz);
    for line in [vec![0], vec![4, 1, 2, 3], vec![5]].iter() {
        let add_offset = || {
            if line.len() == 1 {
                for _ in 0..sz + 2 {
                    eprint!(" ");
                }
            }
        };
        let print_border = || {
            add_offset();
            for _ in 0..(sz + 2) * line.len() {
                eprint!("-");
            }
            eprintln!();
        };
        print_border();
        for r in 0..sz {
            add_offset();
            for &sq_id in line.iter() {
                eprint!("|");
                for c in 0..sz {
                    let x = ids.contains(&squares[sq_id][r][c]);
                    eprint!("{}", if x { "X" } else { "." });
                }
                eprint!("|");
            }
            eprintln!();
        }
        print_border();
    }
}

pub fn calc_num_invs<T: Ord>(a: &[T]) -> usize {
    let mut res = 0;
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            if a[j] < a[i] {
                res += 1;
            }
        }
    }
    res
}

pub fn slice_hash<T: Hash>(a: &[T]) -> u64 {
    let mut hasher = DefaultHasher::new();
    a.hash(&mut hasher);
    hasher.finish()
}

pub fn conv_cube_to_dwalton(sol: &TaskSolution) -> String {
    conv_colors_to_dwalton(&sol.state)
}

pub fn conv_colors_to_dwalton(colors: &[usize]) -> String {
    let conv = |c: char| match c {
        'A' => 'U',
        'B' => 'F',
        'C' => 'R',
        'D' => 'B',
        'E' => 'L',
        'F' => 'D',
        _ => unreachable!(),
    };
    let sz = calc_cube_side_size(colors.len());
    let mut res = String::new();
    let color_names = ['A', 'B', 'C', 'D', 'E', 'F'];
    for &perm in [0, 2, 1, 5, 4, 3].iter() {
        for &color_id in colors[perm * sz * sz..(perm + 1) * sz * sz].iter() {
            let c = conv(color_names[color_id]);
            res.push(c);
        }
    }
    res
}

#[derive(Clone, Debug)]
pub enum DwaltonMove {
    Simple(String),
    Wide(String, RangeInclusive<usize>),
}

fn get_move_index(mv: &str) -> usize {
    let mut res = 0;
    for c in mv.chars() {
        if c.is_ascii_digit() {
            res = res * 10 + c.to_digit(10).unwrap() as usize;
        }
    }
    res
}

impl DwaltonMove {
    pub fn get_index(&self) -> usize {
        match self {
            DwaltonMove::Simple(mv) => get_move_index(mv),
            DwaltonMove::Wide(_mv, r) => {
                if *r.start() == 0 {
                    *r.end()
                } else {
                    *r.start()
                }
            }
        }
    }
}

pub fn conv_dwalton_moves(sz: usize, moves: &str) -> Vec<DwaltonMove> {
    let mut res = vec![];
    let replacements = [
        ("F", "f"),
        ("B", "-f"),
        ("R", "r"),
        ("L", "-r"),
        ("D", "d"),
        ("U", "-d"),
    ];
    for mv in moves.split_ascii_whitespace() {
        let mut found = false;
        for (k, v) in replacements.iter() {
            if let Some((prev, next)) = mv.split_once(k) {
                found = true;
                let mut pos = if prev.is_empty() {
                    0
                } else {
                    prev.parse::<usize>().unwrap() - 1
                };

                let wide = next.contains('w');
                if wide && prev.is_empty() {
                    pos = 1;
                }
                let rev = next.contains('\'');
                let mul2 = next.ends_with('2');
                let rpos = if v.starts_with('-') {
                    sz - 1 - pos
                } else {
                    pos
                };
                // eprintln!("!!! mv={mv} prev={prev} next={next} pos={pos} rpos={rpos} wide={wide} rev={rev} mult2={mul2} sz={sz}");
                if wide {
                    let range = if v.starts_with('-') {
                        rpos..=sz - 1
                    } else {
                        0..=rpos
                    };
                    let v = if rev { rev_move(v) } else { v.to_string() };
                    for _ in 0..(if mul2 { 2 } else { 1 }) {
                        res.push(DwaltonMove::Wide(v.clone(), range.clone()));
                    }
                } else {
                    let mv = format!("{v}{rpos}");
                    let mv = if rev { rev_move(&mv) } else { mv };
                    for _ in 0..(if mul2 { 2 } else { 1 }) {
                        res.push(DwaltonMove::Simple(mv.clone()));
                    }
                }
            }
        }
        assert!(found);
    }
    res
}

#[test]
fn dwalton_moves() {
    let res = conv_dwalton_moves(9, "4Dw' F' R 4Fw2 4Dw2 4Fw' 4Bw 4Uw2 B 4Dw' 4Bw' 3Dw' 3Bw 4Uw2 L2 4Fw2 4Rw' 4Bw2 R2 D' 4Rw 3Uw 3Bw 4Lw2 3Lw2 3Fw' D 3Rw B 3Bw' 3Uw2 3Fw 3Uw' 4Uw2 2Bw 4Rw' 4Bw2 2Uw 4Lw2 4Rw 2Dw' 4Lw2 2Bw' 4Lw' 2Dw2 3Rw2 2Uw' R2 3Uw2 2Dw 3Rw 3Bw2 2Dw2 2Fw2 2Rw' 3Fw2 D2 2Rw2 F' D 2Fw2 2Uw2 2Bw 2Rw' 2Fw R 2Dw' 4Uw2 B 4Bw2 4Rw U' 4Dw2 F' 4Rw D 4Lw' 4Dw2 F' 4Lw2 U' 4Bw2 D 4Rw2 B' 3Rw2 F 4Lw2 3Rw' D' 3Lw' U' B' 3Rw U F 3Uw2 3Lw' 4Dw2 4Rw2 4Bw2 2Lw 4Dw2 F 4Uw2 4Bw2 2Lw' 4Bw2 D 4Bw2 3Fw2 2Bw2 D' 3Rw2 B' 2Lw 3Uw2 2Rw' B 2Rw 3Fw2 B' 2Lw' 2Rw2 U F 2Bw2 D2 2Rw 2Dw2 2Lw' 4Dw2 R 4Dw2 4Fw2 3Bw2 4Dw2 4Bw2 R' 3Bw2 4Dw2 4Bw2 4Lw2 U 3Lw2 U' 4Fw2 3Bw2 4Lw2 4Bw2 3Lw2 U 4Fw2 B2 4Uw2 4Lw2 4Uw2 3Rw2 F' 4Lw2 B 3Rw2 4Uw2 F' B2 3Rw2 4Fw2 R' 4Fw2 4Uw2 2Dw2 4Rw2 4Uw2 2Bw2 R' 2Dw2 U' 4Bw2 2Lw2 4Bw2 U 2Bw2 4Lw2 U' B 4Dw2 2Lw2 F2 4Uw2 2Dw2 F' R2 4Uw2 F 2Rw2 4Dw2 F' 4Dw2 2Dw2 3Dw2 R 3Uw2 3Dw2 R' 2Uw2 3Uw2 3Rw2 3Uw2 L' 2Rw2 2Bw2 3Rw2 3Bw2 2Rw2 U2 3Lw2 U' 3Rw2 3Bw2 U' B2 2Uw2 3Rw2 B 3Uw2 2Rw2 3Rw2 F2 3Lw2 3Dw2 F 2Rw2 2Dw2 3Dw2 B L U' 4Dw2 F' 4Dw2 R' U' D' B D2 B' 4Uw2 4Rw2 D2 4Dw2 4Lw2 D2 F2 B 4Dw2 4Rw2 U' F2 U 4Bw2 U' B2 D 4Rw2 4Fw2 4Bw2 4Rw2 B2 L2 4Lw2 4Bw2 L 3Rw2 F 3Rw2 U' 3Rw2 D L' B' 3Uw2 3Dw2 L2 B2 3Lw2 R2 B 3Uw2 3Lw2 F2 L2 U2 F 3Dw2 3Fw2 D F2 L2 D R2 3Bw2 L2 3Lw2 3Fw2 U 3Fw2 3Bw2 U2 3Bw2 2Uw2 R 2Fw2 R B' U' 2Uw2 L' L2 D 2Fw2 D2 2Lw2 D2 B' 2Dw2 R2 2Uw2 2Rw2 B 2Lw2 R2 B D' 2Bw2 2Rw2 F2 B2 U' F2 U2 2Fw2 D 2Rw2 B2 2Rw2 2Bw2 D 2Bw2 F' U' F' R' U2 F2 R' B2 U R U' L2 F2 U' R2 U F2 R2 L2");
    eprintln!("res={:?}", res);
}

#[test]
fn dwalton_moves4() {
    let res = conv_dwalton_moves(4, "Rw' B2 Rw2 U2 Rw' Dw2 R F2 Lw' L2 U Rw2 Uw2 D2 B Dw2 B2 L2 D' Lw2 U2 L2 D' L2 Lw2 D R2 Fw2 D' F L' B2 D' L2 F' U F R B2 R2 B2 U R2 F2 B2 U R2 B2");
    eprintln!("res={:?}", res);
}
