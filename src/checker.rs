use crate::puzzle::Puzzle;

pub fn check_solution(task: &Puzzle, solution: &[String]) {
    let mut state = task.initial_state.clone();
    for step in solution.iter() {
        let perm = &task.info.moves[step];
        for cycle in perm.cycles.iter() {
            for w in cycle.windows(2) {
                state.swap(w[0], w[1]);
            }
        }
    }
    let mut cnt_fails = 0;
    for i in 0..state.len() {
        if state[i] != task.solution_state[i] {
            cnt_fails += 1;
        }
    }
    println!("{}: {}/{} fails", task.id, cnt_fails, task.num_wildcards);
    assert!(cnt_fails <= task.num_wildcards)
}
