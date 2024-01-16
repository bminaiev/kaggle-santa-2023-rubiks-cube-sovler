use std::fs::{File, OpenOptions};

use crate::{data::Data, sol_utils::TaskSolution};

pub struct Event {
    pub task_id: usize,
    pub solution: Vec<String>,
}

impl Event {
    pub fn to_task(&self, data: &Data) -> TaskSolution {
        let mut task = TaskSolution::new(data, self.task_id);
        for mv in self.solution.iter() {
            task.append_move(mv);
        }
        task
    }
}

pub struct SolutionsLog {
    f: File,
    pub events: Vec<Event>,
}

const FILE_PATH: &str = "data/solutions.log";

impl SolutionsLog {
    pub fn new() -> Self {
        let mut events = vec![];
        if std::path::Path::new(FILE_PATH).exists() {
            let mut reader = csv::Reader::from_path(FILE_PATH).unwrap();

            for result in reader.records() {
                let record = result.unwrap();
                let id = record[0].parse::<usize>().unwrap();
                let solution: Vec<String> = record[1].split('.').map(|s| s.to_string()).collect();
                events.push(Event {
                    task_id: id,
                    solution,
                });
            }
        } else {
            File::create(FILE_PATH).unwrap();
        }
        let f = OpenOptions::new()
            .write(true)
            .append(true)
            .open(FILE_PATH)
            .unwrap();

        Self { f, events }
    }

    pub fn append(&mut self, task: &TaskSolution) {
        use std::io::Write;
        self.events.push(Event {
            task_id: task.task_id,
            solution: task.answer.clone(),
        });
        writeln!(self.f, "{},{}", task.task_id, task.answer.join(".")).unwrap();
        self.f.flush().unwrap();
        eprintln!("SAVED SOLUTION FOR TASK {}", task.task_id);
    }
}
