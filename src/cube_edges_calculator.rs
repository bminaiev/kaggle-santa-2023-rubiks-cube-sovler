pub fn calc_cube_edges(squares: &[Vec<Vec<usize>>]) -> Vec<Vec<[usize; 2]>> {
    let n = squares[0].len();
    // 3 -> 0
    // 4 -> 1
    // 5 -> 1
    // 6 -> 2
    let mut res = vec![vec![]; (n - 2) / 2];
    // 3 -> (1, 1)
    // 4 -> (1, 2)
    let center1 = (n - 1) / 2;
    let center2 = n / 2;

    let is_corner =
        |r: usize, c: usize| -> bool { (r == 0 || r == n - 1) && (c == 0 || c == n - 1) };

    let on_edge = |r: usize, c: usize| -> bool {
        if is_corner(r, c) {
            return false;
        }
        r == 0 || r == n - 1 || c == 0 || c == n - 1
    };

    let dist_to_center = |r: usize, c: usize| -> usize {
        if r == 0 || r == n - 1 {
            if c <= center1 {
                center1 - c
            } else {
                c - center2
            }
        } else if r <= center1 {
            center1 - r
        } else {
            r - center2
        }
    };

    for r in 0..n {
        for c in 0..n {
            for dr in 0..=1 {
                let dc = 1 - dr;
                if r + dr < n && c + dc < n {
                    let nr = r + dr;
                    let nc = c + dc;
                    if on_edge(r, c) && on_edge(nr, nc) {
                        let dist1 = dist_to_center(r, c);
                        let dist2 = dist_to_center(nr, nc);
                        for sq in squares.iter() {
                            if dist1 < dist2 {
                                res[dist1].push([sq[r][c], sq[nr][nc]]);
                            } else {
                                res[dist2].push([sq[nr][nc], sq[r][c]]);
                            }
                        }
                    }
                }
            }
        }
    }
    res
}

pub fn build_squares(sz: usize) -> Vec<Vec<Vec<usize>>> {
    let mut res = vec![];
    for i in 0..6 {
        let offset = i * sz * sz;
        let mut sq = vec![vec![0; sz]; sz];
        for r in 0..sz {
            for c in 0..sz {
                sq[r][c] = offset + r * sz + c;
            }
        }
        res.push(sq);
    }
    res
}

pub fn calc_edges_score(edges: &[Vec<[usize; 2]>], state: &[usize]) -> Vec<usize> {
    let norm = |a: [usize; 2]| {
        if a[0] < a[1] {
            a
        } else {
            [a[1], a[0]]
        }
    };
    edges
        .iter()
        .map(|edges| {
            let mut real_neigh = edges
                .iter()
                .map(|edge| norm([state[edge[0]], state[edge[1]]]))
                .collect::<Vec<_>>();
            real_neigh.sort();
            let mut cnt_ok = 0;
            for edge in edges.iter() {
                if real_neigh.binary_search(&norm(*edge)).is_ok() {
                    cnt_ok += 1;
                }
            }
            assert!(cnt_ok % 2 == 0);
            cnt_ok / 2
        })
        .collect()
}

pub fn calc_cube_centers(squares: &[Vec<Vec<usize>>]) -> Vec<bool> {
    let n = squares[0].len();
    let mut res = vec![true; 6 * n * n];
    for sq in squares.iter() {
        for r in 0..n {
            for c in 0..n {
                if r == 0 || r == n - 1 || c == 0 || c == n - 1 {
                    res[sq[r][c]] = false;
                }
            }
        }
    }
    res
}

#[test]
fn test() {
    let sz = 3;
    let squares = build_squares(sz);
    let edges = calc_cube_edges(&squares);
    for lvl in 0..edges.len() {
        eprintln!("lvl={}", lvl);
        for edge in edges[lvl].iter() {
            eprintln!("{} {}", edge[0], edge[1]);
        }
    }
}
