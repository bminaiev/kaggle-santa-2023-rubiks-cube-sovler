use crate::{
    cube_edges_calculator::build_squares,
    data::Data,
    kociemba_solver::kociemba_solve,
    puzzle::Puzzle,
    sol_utils::TaskSolution,
    solver3::Solver3,
    utils::{calc_cube_side_size, show_cube_ids},
};

pub struct Cube3Converter {
    cube3_solver: Solver3,
}

impl Cube3Converter {
    pub fn new(cube3_solver: Solver3) -> Self {
        Self { cube3_solver }
    }

    pub fn solve(&self, data: &Data, task: &mut TaskSolution, exact_perm: bool) -> bool {
        let n = task.task.info.n;
        let sz = calc_cube_side_size(n);
        let squares = build_squares(sz);
        let ids_groups = squares
            .iter()
            .flat_map(|s| {
                let mut res = vec![];
                let splits = [0, 1, sz - 1, sz];
                for wx in splits.windows(2) {
                    for wy in splits.windows(2) {
                        let mut cur = vec![];
                        for x in wx[0]..wx[1] {
                            for y in wy[0]..wy[1] {
                                cur.push(s[x][y]);
                            }
                        }
                        res.push(cur);
                    }
                }
                res
            })
            .collect::<Vec<_>>();
        assert_eq!(ids_groups.len(), 54);
        let mut group_id = vec![usize::MAX; n];
        for i in 0..ids_groups.len() {
            for &x in ids_groups[i].iter() {
                group_id[x] = i;
            }
        }
        let mut state = vec![usize::MAX; ids_groups.len()];
        for i in 0..state.len() {
            for &x in ids_groups[i].iter() {
                let expected_group = task.state[x];
                assert!(state[i] == usize::MAX || state[i] == expected_group);
                state[i] = expected_group;
            }
        }
        let answer = if task.exact_perm {
            let fake_task = Puzzle {
                id: usize::MAX,
                puzzle_type: "cube_3/3/3".to_owned(),
                solution_state: (0..state.len()).collect(),
                initial_state: state.clone(),
                num_wildcards: 0,
                num_colors: 6,
                color_names: ["A", "B", "C", "D", "E", "F"]
                    .iter()
                    .map(|&x| x.to_owned())
                    .collect(),
                info: data.puzzle_info.get("cube_3/3/3").unwrap().clone(),
            };
            let mut new_task = TaskSolution::new_fake(state, fake_task);
            if exact_perm {
                for _it in 0..10 {
                    eprintln!("TRY SOLVING... {_it}");
                    let mut task_copy = new_task.clone();
                    if self.cube3_solver.solve_task_with_rotations(&mut task_copy) {
                        eprintln!("WOW! Solved cube3 with permutations...");
                        new_task = task_copy;
                        break;
                    }
                }
            } else {
                self.cube3_solver.solve_task(&mut new_task);
            }
            new_task.answer
        } else {
            eprintln!("State: {state:?}");
            let colors = [b'U', b'F', b'R', b'B', b'L', b'D'];
            let squares_perm = [0, 2, 1, 5, 4, 3];
            let mut kociemba = vec![];
            for &sq_id in squares_perm.iter() {
                for r in 0..3 {
                    for c in 0..3 {
                        let our_pos = sq_id * 9 + r * 3 + c;
                        let our_color = colors[state[our_pos]];
                        kociemba.push(our_color);
                    }
                }
            }
            let kociemba = String::from_utf8(kociemba).unwrap();
            eprintln!("STATE TO SOLVE: {kociemba}");
            let kociemba_moves = kociemba_solve(&kociemba);
            eprintln!("MOVES: {kociemba_moves:?}");
            match kociemba_moves {
                Some(kociemba_moves) => kociemba_moves,
                None => {
                    eprintln!("KOCIEMBA FAILED!");
                    return false;
                }
            }
        };

        for mv in answer.iter() {
            let mv = if mv.ends_with('2') {
                format!("{}{}", &mv[..mv.len() - 1], sz - 1)
            } else {
                assert!(mv.ends_with('0'));
                mv.clone()
            };
            task.append_move(&mv);
        }

        task.print(data);
        show_cube_ids(&task.get_correct_colors_positions(), sz);

        assert!(task.is_solved());
        true
    }
}
