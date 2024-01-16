use crate::{data::Data, sol_utils::TaskSolution, solutions_log::SolutionsLog};

pub fn solve_wreath(data: &Data, log: &mut SolutionsLog) {
    let mut tasks = TaskSolution::all_by_type(data, "wreath", false);

    for task in tasks.iter() {
        eprintln!("Task {}. Type: {}", task.task_id, task.task.puzzle_type);
        for (k, v) in task.task.info.moves.iter() {
            eprintln!("{} -> {:?}", k, v);
        }
    }
}
