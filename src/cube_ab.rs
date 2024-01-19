use crate::{
    cube_edges_calculator::build_squares, data::Data, dsu::Dsu, sol_utils::TaskSolution,
    utils::calc_cube_side_size,
};
use std::io::Write;

pub fn cube_ab_solver(data: &Data) {
    let tasks = TaskSolution::all_by_type(data, "cube_", false);
    let tasks: Vec<_> = tasks
        .into_iter()
        .filter(|t| t.task.get_color_type() == "B")
        .collect();

    let mut f = std::fs::File::create("data/puzzles_ab.csv").unwrap();
    writeln!(
        f,
        "id,puzzle_type,solution_state,initial_state,num_wildcards"
    )
    .unwrap();
    for t in tasks.iter() {
        eprintln!("Task: {}. Color: {}", t.task.id, t.task.get_color_type());

        let puzzle_info = &t.task.info;
        let n = puzzle_info.n;
        eprintln!("n={n}");
        let sz = calc_cube_side_size(n);
        if sz % 2 == 0 {
            continue;
        }

        let mut dsu = Dsu::new(n);
        for (_k, perm) in puzzle_info.moves.iter() {
            for cycle in perm.cycles.iter() {
                for w in cycle.windows(2) {
                    dsu.unite(w[0], w[1]);
                }
            }
        }

        for group in dsu.get_groups() {
            eprintln!("Group: {:?}", group);
            let mut target_colors = vec![];
            for &v in group.iter() {
                target_colors.push(t.task.solution_state[v]);
            }
            eprintln!("Color: {target_colors:?}");
        }

        let conv = |a: &[usize]| {
            let res: Vec<_> = a
                .iter()
                .map(|x| {
                    let square = x / sz / sz;
                    String::from_utf8(vec![b'A' + square as u8]).unwrap()
                })
                .collect();
            res.join(";")
        };

        let solution_state = conv((0..n).collect::<Vec<_>>().as_slice());
        let initial_state = conv(t.state.as_slice());
        writeln!(
            f,
            "{},{},{},{},{}",
            t.task.id, t.task.puzzle_type, solution_state, initial_state, t.task.num_wildcards
        )
        .unwrap();
    }
}
