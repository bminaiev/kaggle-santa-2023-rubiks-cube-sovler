use std::process::Command;

use crate::moves::rev_move;

pub fn kociemba_solve(state: &str) -> Option<Vec<String>> {
    let output = Command::new("kociemba")
        .arg(state)
        .output()
        .expect("failed to execute process");
    let output = String::from_utf8(output.stdout).unwrap();
    eprintln!("kociemba output: {}", output);
    if output.starts_with("ERROR") {
        return None;
    }
    let moves = output.split_ascii_whitespace().collect::<Vec<_>>();
    eprintln!("kociemba moves: {:?}", moves);
    let replacements = [
        ("F", "f0"),
        ("B", "-f2"),
        ("R", "r0"),
        ("L", "-r2"),
        ("D", "d0"),
        ("U", "-d2"),
    ];
    let mut res = vec![];
    for mv in moves.iter() {
        let mut mv = mv.to_string();
        let mut rev = false;
        if mv.ends_with('\'') {
            mv.pop();
            rev = true;
        }
        let mut x2 = false;
        if mv.ends_with('2') {
            mv.pop();
            x2 = true;
        }
        let mut new_move = String::new();
        for (old, new) in replacements.iter() {
            if mv == *old {
                new_move = new.to_string();
                break;
            }
        }
        assert!(!new_move.is_empty());
        if rev {
            new_move = rev_move(&new_move);
        }
        res.push(new_move.to_string());
        if x2 {
            res.push(new_move.to_string());
        }
    }
    Some(res)
}

#[test]
fn test() {
    let state = "BFRRUFRRUFLUDRULUBDDLBFFBDDLLFBDRULRRBFRLDBUDFUULBFDBL";
    let solution = kociemba_solve(state);
    eprintln!("solution={:?}", solution);
}

// Expected fail..
#[test]
fn test2() {
    let state = "RBRRUFLRLUUBDRUBBUFUFRFFDDDRRLFDLUDBDBDBLFFLBUDFLBULLR";
    let solution = kociemba_solve(state);
    eprintln!("solution={:?}", solution);
}
