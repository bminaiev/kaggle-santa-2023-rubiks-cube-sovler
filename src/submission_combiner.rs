use std::{collections::HashMap, fs::File};

use crate::{data::Data, sol_utils::TaskSolution, solutions_log::SolutionsLog};

// https://www.kaggle.com/competitions/santa-2023/discussion/466500
fn score_200k_estimator(puzzle_type: &str, data: &Data) -> Option<usize> {
    let cnt = data
        .puzzles
        .iter()
        .filter(|p| p.puzzle_type == puzzle_type)
        .count();
    Some(
        match puzzle_type {
            "cube_2/2/2" => 300,
            "cube_3/3/3" => 3000,
            "cube_4/4/4" => 7000,
            "cube_5/5/5" => 5000,
            "cube_6/6/6" => 7000,
            "cube_7/7/7" => 2000,
            "cube_8/8/8" => 4000,
            "cube_9/9/9" => 5000,
            "cube_10/10/10" => 10_000,
            "cube_19/19/19" => 20_000,
            "cube_33/33/33" => 80_000,
            _ => return None,
        } / cnt,
    )
}

fn get_task_type(s: &str) -> &str {
    if s.starts_with("cube") {
        return "cube";
    }
    if s.starts_with("globe") {
        return "globe";
    }
    if s.starts_with("wreath") {
        return "wreath";
    }
    unreachable!();
}

fn save_submission(tasks: &[TaskSolution]) {
    let cost = tasks.iter().map(|t| t.answer.len()).sum::<usize>();
    eprintln!("Total cost: {}", cost);

    let mut hm = HashMap::<String, usize>::new();
    for task in tasks.iter() {
        assert!(task.is_solved_with_wildcards());

        let task_name = format!(
            "{}_{}",
            get_task_type(&task.task.puzzle_type),
            task.task.get_color_type()
        );
        *hm.entry(task_name).or_default() += task.answer.len();
    }
    for (k, v) in hm.iter() {
        println!("{} -> {}", k, v);
    }

    use std::io::Write;
    let name = format!("data/my_{}k.csv", cost / 1000);
    let mut f = File::create(&name).unwrap();
    writeln!(f, "id,moves").unwrap();
    for task in tasks.iter() {
        writeln!(f, "{},{}", task.task_id, task.answer.join(".")).unwrap();
    }

    let mut f_log = File::create(name + ".txt").unwrap();
    for task in tasks.iter() {
        writeln!(
            f_log,
            "{}\t{}({})\t{}",
            task.task_id,
            task.task.puzzle_type,
            task.task.get_color_type(),
            task.answer.len()
        )
        .unwrap();
    }
}

pub fn make_submission(data: &Data, log: &SolutionsLog) {
    for task in data.puzzles.iter() {
        let sample_sol_len = data.solutions.sample[&task.id].len();
        let s853k_sol_len = data.solutions.s853k[&task.id].len();
        let s200k_est = score_200k_estimator(&task.puzzle_type, data).unwrap_or_default();
        let color_type = task.get_color_type();
        let task_name = format!("{}_{}", get_task_type(&task.puzzle_type), color_type);

        eprintln!(
            "Task {}. Type: {}. Colors: {color_type}. Sample len: {sample_sol_len}. 853k: {s853k_sol_len}. 200k: {s200k_est}",
            task.id,
            task.puzzle_type,
        );
    }

    let mut tasks: Vec<_> = TaskSolution::from_solutions_file(data, &data.solutions.s853k);
    for i in 0..tasks.len() {
        assert_eq!(tasks[i].task_id, i);
    }
    let calc_scores =
        |tasks: &[TaskSolution]| -> usize { tasks.iter().map(|t| t.answer.len()).sum::<usize>() };
    eprintln!("Sum scores: {}", calc_scores(&tasks));
    let mut events_perm: Vec<_> = (0..log.events.len()).collect();
    events_perm.sort_by_key(|&i| (log.events[i].task_id, log.events[i].solution.len()));
    for idx in events_perm.into_iter() {
        let event = &log.events[idx];
        let task = event.to_task(data);
        if !task.is_solved_with_wildcards() {
            continue;
        }
        let task_id = task.task_id;
        if task.answer.len() < tasks[task.task_id].answer.len() {
            eprintln!(
                "Wow! Better solution for task {} ({}, {}). {} -> {}",
                task.task_id,
                task.task.puzzle_type,
                task.task.get_color_type(),
                tasks[task.task_id].answer.len(),
                task.answer.len()
            );
            tasks[task_id] = task;
        }
    }
    let tasks_ab = TaskSolution::from_solutions_file(data, &data.solutions.ab);
    for id in 0..tasks.len() {
        if tasks_ab[id].answer.len() < tasks[id].answer.len() {
            eprintln!(
                "Wow! Better AB solution for task {} ({}, {}). {} -> {}",
                id,
                tasks[id].task.puzzle_type,
                tasks[id].task.get_color_type(),
                tasks[id].answer.len(),
                tasks_ab[id].answer.len()
            );
            tasks[id] = tasks_ab[id].clone();
        }
    }
    let mut cur_sum_len: usize = calc_scores(&tasks);
    const TARGET_SCORE: usize = 133_333;
    for id in 0..tasks.len() {
        if tasks[id].task.num_wildcards > 0 {
            let mut task_tmp = tasks[id].clone();
            task_tmp.reset(data);
            for i in 0..tasks[id].answer.len() - 1 {
                task_tmp.append_move(&tasks[id].answer[i]);
                if task_tmp.is_solved_with_wildcards() {
                    let delta = tasks[id].answer.len() - i - 1;
                    if cur_sum_len - delta >= TARGET_SCORE {
                        cur_sum_len -= delta;
                        eprintln!(
                            "WOW! CAN TRUNCATE TASK {id}: {} -> {}.",
                            tasks[id].answer.len(),
                            i + 1
                        );
                        tasks[id].answer.truncate(i + 1);
                        break;
                    }
                }
            }
        }
    }
    eprintln!("New sum scores: {}", calc_scores(&tasks));
    save_submission(&tasks);
}
