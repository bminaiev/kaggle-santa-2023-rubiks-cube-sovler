pub struct Dsu {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Dsu {
    pub fn new(n: usize) -> Self {
        let mut parent = vec![0; n];
        for i in 0..n {
            parent[i] = i;
        }
        Dsu {
            parent,
            size: vec![1; n],
        }
    }

    pub fn get(&mut self, x: usize) -> usize {
        if self.parent[x] == x {
            x
        } else {
            let p = self.parent[x];
            self.parent[x] = self.get(p);
            self.parent[x]
        }
    }

    pub fn unite(&mut self, x: usize, y: usize) {
        let px = self.get(x);
        let py = self.get(y);
        if px != py {
            self.parent[px] = py;
            self.size[py] += self.size[px];
        }
    }

    pub fn get_size(&mut self, x: usize) -> usize {
        let px = self.get(x);
        self.size[px]
    }

    pub fn get_groups(&mut self) -> Vec<Vec<usize>> {
        let mut res = vec![vec![]; self.parent.len()];
        for i in 0..self.parent.len() {
            res[self.get(i)].push(i);
        }
        res.into_iter().filter(|x| x.len() > 1).collect()
    }
}
