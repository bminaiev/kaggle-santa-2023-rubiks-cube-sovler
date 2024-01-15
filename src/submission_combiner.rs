use crate::data::Data;

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

pub fn make_submission(data: &Data) {
    for task in data.puzzles.iter() {
        let sample_sol_len = data.solutions.sample[&task.id].len();
        let s853k_sol_len = data.solutions.s853k[&task.id].len();
        let s200k_est = score_200k_estimator(&task.puzzle_type, data).unwrap_or_default();
        let color_type = task.get_color_type();
        eprintln!(
            "Task {}. Type: {}. Colors: {color_type}. Sample len: {sample_sol_len}. 853k: {s853k_sol_len}. 200k: {s200k_est}",
            task.id,
            task.puzzle_type,
        );
    }
}
