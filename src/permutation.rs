use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Permutation {
    pub cycles: Vec<Vec<usize>>,
}

impl Permutation {
    pub fn identity() -> Self {
        Self { cycles: vec![] }
    }

    pub fn sum_len(&self) -> usize {
        self.cycles.iter().map(|c| c.len()).sum()
    }

    pub fn from_array(p: &[usize]) -> Self {
        let mut cycles = vec![];
        let mut seen = vec![false; p.len()];
        for start in 0..p.len() {
            let mut cycle = vec![];
            let mut cur = start;
            while !seen[cur] {
                cycle.push(cur);
                seen[cur] = true;
                cur = p[cur];
            }
            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }
        Self { cycles }
    }

    pub fn x2(&self) -> Self {
        self.combine(self)
    }

    pub fn apply<T>(&self, a: &mut [T]) {
        for cycle in self.cycles.iter() {
            for w in cycle.windows(2) {
                a.swap(w[0], w[1]);
            }
        }
    }

    pub fn apply_rev(&self, a: &mut [usize]) {
        for cycle in self.cycles.iter() {
            for w in cycle.windows(2).rev() {
                a.swap(w[0], w[1]);
            }
        }
    }

    pub fn conv_to_array(&self, n: usize) -> Vec<usize> {
        let mut a = vec![0; n];
        for i in 0..n {
            a[i] = i;
        }
        for cycle in self.cycles.iter() {
            for i in 0..cycle.len() {
                a[cycle[i]] = cycle[(i + 1) % cycle.len()];
            }
        }
        a
    }

    pub fn combine_linear(&self, other: &Self) -> Self {
        let mut n = 0;
        for perm in [self, other].iter() {
            for cycle in perm.cycles.iter() {
                for x in cycle.iter() {
                    n = n.max(*x);
                }
            }
        }
        let p1 = self.conv_to_array(n + 1);
        let p2 = other.conv_to_array(n + 1);
        Self::from_array(&p2.iter().map(|&x| p1[x]).collect::<Vec<_>>())
    }

    pub fn combine(&self, other: &Self) -> Self {
        let mut a = HashMap::new();
        for who in [self, other].iter() {
            for cycle in who.cycles.iter() {
                for w in cycle.windows(2) {
                    let v1 = *a.get(&w[0]).unwrap_or(&w[0]);
                    let v2 = *a.get(&w[1]).unwrap_or(&w[1]);
                    a.insert(w[0], v2);
                    a.insert(w[1], v1);
                }
            }
        }
        let mut cycles = vec![];
        let mut seen = HashSet::new();
        let mut all_keys: Vec<usize> = a.keys().cloned().collect();
        all_keys.sort();
        for &start in all_keys.iter() {
            let mut cycle = vec![];
            let mut cur = start;
            while !seen.contains(&cur) {
                cycle.push(cur);
                seen.insert(cur);
                cur = a[&cur];
            }
            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }
        Self { cycles }
    }

    pub fn inv(&self) -> Self {
        let cycles = self
            .cycles
            .iter()
            .map(|cycle| cycle.iter().rev().cloned().collect())
            .collect();
        Self { cycles }
    }

    // TODO: optimize
    pub fn next(&self, x: usize) -> usize {
        for cycle in self.cycles.iter() {
            for i in 0..cycle.len() {
                if cycle[i] == x {
                    // TODO: check?
                    return cycle[(i + cycle.len() - 1) % cycle.len()];
                }
            }
        }
        x
    }
}
