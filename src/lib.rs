use std::collections::HashMap;
use std::cmp::{Eq, PartialEq};
use std::fmt;

// TODO: test if is valid initial board
#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Board([[usize; 3]; 3]);

impl Board {
  pub fn new(board: [[usize; 3]; 3]) -> Board {
    Board(board)
  }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}\n{} {} {}\n{} {} {}\n", 
            self.0[0][0], self.0[0][1], self.0[0][2],
            self.0[1][0], self.0[1][1], self.0[1][2],
            self.0[2][0], self.0[2][1], self.0[2][2]
        )
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct State {
    pub board: Board,
    moves: usize,
    cost: usize,
    came_from: Option<Vector2>
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
struct Vector2(usize, usize);

impl State {
    fn new(board: Board, moves: usize, cost: usize, came_from: Option<Vector2>) -> State {
        State { board, moves, cost , came_from }
    }

    fn next_states(&self, goal: &Board, heuristic: &dyn Fn(&Board, &Board) -> usize) -> Vec<State> {        
        let empty_space = self.empty_space();
        let mut neighbors = State::neighbors(empty_space);
        if let Some(came_from) = self.came_from {
            neighbors.retain(|&x| x != came_from);
        }
        let states: Vec<State> = neighbors.iter().map(|&x| {
           let moves = self.moves + 1;
           let board = self.move_empty_space(x, empty_space);
           let cost = heuristic(&goal, &board) + moves;
           State { board, moves, cost, came_from: Some(empty_space) }
        }).collect();
        states
    }
    
    fn move_empty_space(&self, to: Vector2, empty_space: Vector2) -> Board {
        let mut board = self.board.clone();
        board.0[empty_space.0][empty_space.1] = board.0[to.0][to.1];
        board.0[to.0][to.1] = 0;
        board
    }

    fn neighbors(empty_space: Vector2) -> Vec<Vector2> {
        let mut result: Vec<Vector2> = vec![];
        if empty_space.0 > 0 {
            result.push(Vector2(empty_space.0 - 1, empty_space.1))
        }
        if empty_space.0 < 2 {
            result.push(Vector2(empty_space.0 + 1, empty_space.1))
        }
        if empty_space.1 > 0 {
            result.push(Vector2(empty_space.0, empty_space.1 - 1))
        }
        if empty_space.1 < 2 {
            result.push(Vector2(empty_space.0, empty_space.1 + 1))
        }
        result
    }

    fn empty_space(&self) -> Vector2 {
        for i in 0..3 {
            for j in 0..3 {
                if self.board.0[i][j] == 0 {
                    return Vector2(i,j);
                }        
            }
        }
        Vector2(0,0)
    }
}

#[derive(Debug)]
pub struct ArenaTree<T> where T: PartialEq + Clone {
    pub arena: Vec<Node<T>>
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node<T> where T: PartialEq + Clone {
    index: usize,
    pub val: T,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub depth: usize,
    pub x: f32,
    thread: Option<usize>,
    module: f32,
    ancestor: usize,
    change: f32,
    shift: f32,
    pub number: usize
}

#[derive(Clone)]
pub struct NodeVisualizer<'a, T> where T: PartialEq + Clone {
    pub node: &'a Node<T>,
    pub x: usize
}

impl<T> Node<T> where T: PartialEq + Clone {
    fn new(index: usize, val: T, parent: Option<usize>, depth: usize, number: usize) -> Self {
        Self { index, val, parent, children: vec![], depth, x: 0.0, thread: None, ancestor: index, change: 0.0, shift: 0.0, module: 0.0, number }
    }
}

impl<T> ArenaTree<T> where T: PartialEq + Clone {
    fn new() -> Self {
        ArenaTree { arena: vec![] }
    }

    fn get_index(&self, val: &T) -> Option<usize> {
        for node in &self.arena {
            if &node.val == val {
                return Some(node.index);
            }
        }
        None
    }

    pub fn get_node(&self, index: usize) -> Option<&Node<T>> {
        if index < self.arena.len() {
            return Some(&self.arena[index]);
        }
        None
    }

    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut Node<T>> {
        if index < self.arena.len() {
            return Some(&mut self.arena[index]);
        }
        None
    }
    
    fn insert(&mut self, val: T, parent_index: Option<usize>) {
        let index = self.arena.len();
        let mut depth = 0;
        let mut number = 1;
        if let Some(index) = parent_index {
            let parent_node = self.get_node(index).expect("Cannot insert node with invalid parent");
            number = parent_node.children.len() + 1;
            depth = parent_node.depth + 1;
        }
        self.arena.push(Node::new(index, val, parent_index, depth, number));
        if let Some(parent_index) = parent_index {
            self.arena[parent_index].children.push(index);
        }
    }

    fn left(&self, index: usize) -> Option<usize> {
        if let Some(node) = self.get_node(index) {
            if node.children.len() > 0 {
                return Some(node.children[0]);
            }
            if node.thread != None {
                return node.thread;
            }
        }
        None
    }

    fn right(&self, index: usize) -> Option<usize> {
        if let Some(node) = self.get_node(index) {
            let len = node.children.len();
            if len > 0 {
                return Some(node.children[len - 1]);
            }
            if node.thread != None {
                return node.thread;
            }
        }
        None
    }

    pub fn buchheim(&mut self) {
        self.first_walk(1.0);
        for i in 0..self.arena.len() {
            self.arena[i].module = 0.0;
            self.arena[i].thread = Some(0);
            self.arena[i].ancestor = i;
        }
        let min = self.second_walk();
        if min < 0.0 {
            self.third_walk(-min);
        }
    }

    fn first_walk(&mut self, distance: f32) {
        #[derive(Debug)]
        struct Snapshot {
            index: usize,
            stage: usize,
            default_ancestor: usize,
        }
        let mut stack: Vec<Snapshot> = Vec::new();
        if self.arena.len() > 0 {
            stack.push(Snapshot { index: 0, stage: 0, default_ancestor: 0 });
        }
        while !stack.is_empty() {
            println!("\n\n{:?}\n\n", stack);
            let s = stack.pop().unwrap();
            let node_index = s.index;
            let stage = s.stage;
            let default_ancestor = s.default_ancestor;
            match stage {
                0 => {
                    if self.arena[node_index].children.len() > 0 {
                        stack.push(Snapshot { index: node_index, stage: 2, default_ancestor: 0 });
                    }
                    if self.arena[node_index].children.len() == 0 {
                        if self.leftmost_sibling(node_index) != None {
                            let left_brother = self.left_brother(node_index).unwrap();
                            self.arena[node_index].x = self.arena[left_brother].x + distance;
                        } else {
                            self.arena[node_index].x = 0.0;
                        }
                    } else {
                        let default_ancestor = self.arena[node_index].children[0];
                        stack.push(Snapshot { index: self.arena[node_index].children[0], stage: 1, default_ancestor });
                        stack.push(Snapshot { index: self.arena[node_index].children[0], stage: 0, default_ancestor });
                        // for &child_index in &self.arena[node_index].children.clone() {
                        // }
                    }
                },
                1 => {
                    let default_ancestor = self.apportion(node_index, default_ancestor, distance);
                    if let Some(right) = self.right(node_index) {
                        stack.push(Snapshot { index: right, stage: 1, default_ancestor });
                        stack.push(Snapshot { index: right, stage: 0, default_ancestor });
                    } 
                },
                2 => {
                    self.execute_shifts(node_index);

                    let first_child = self.arena[node_index].children[0];

                    let len = self.arena[node_index].children.len() - 1;
                    let last_child = self.arena[node_index].children[len];
                    let midpoint = (self.arena[first_child].x + self.arena[last_child].x) / 2.0;

                    if let Some(left_brother) = self.left_brother(node_index) {
                        self.arena[node_index].x = self.arena[left_brother].x + distance;
                        self.arena[node_index].module = self.arena[node_index].x - midpoint;
                    } else {
                        self.arena[node_index].x = midpoint;
                    }
                },
                _ => ()
            }
            
            // if self.arena[node_index].children.len() == 0 {
            //     if self.leftmost_sibling(node_index) != None {
            //         let left_brother = self.left_brother(node_index).unwrap();
            //         self.arena[node_index].x = self.arena[left_brother].x + distance;
            //     } else {
            //         self.arena[node_index].x = 0.0;
            //     }
            // } else {
            //     let mut default_ancestor = self.arena[node_index].children[0];
            //     for &child_index in &self.arena[node_index].children.clone() {
            //         stack.push((child_index, 0));
            //         default_ancestor = self.apportion(child_index, default_ancestor, distance);
            //     }
            //     self.execute_shifts(node_index);

            //     let first_child = self.arena[node_index].children[0];

            //     let len = self.arena[node_index].children.len() - 1;
            //     let last_child = self.arena[node_index].children[len];
            //     let midpoint = (self.arena[first_child].x + self.arena[last_child].x) / 2.0;

            //     if let Some(left_brother) = self.left_brother(node_index) {
            //         self.arena[node_index].x = self.arena[left_brother].x + distance;
            //         self.arena[node_index].module = self.arena[node_index].x - midpoint;
            //     } else {
            //         self.arena[node_index].x = midpoint;
            //     }
            // }
        }
    }

    fn second_walk(&mut self) -> f32 {
        let mut stack: Vec<(usize, f32)> = Vec::new();
        let mut min: f32 = 0.0;
        if self.arena.len() > 0 {
            min = self.arena[0].x;
            stack.push((0, 0.0));
        }

        while !stack.is_empty() {
            let s = stack.pop().unwrap();
            let node_index = s.0;
            let m = s.1;
            self.arena[node_index].x += m;
            
            if self.arena[node_index].x < min {
                min = self.arena[node_index].x;
            }
            
            for &child in &self.arena[node_index].children {
                stack.push((child, m + self.arena[node_index].module));
            }
        }
        min
    }

    fn third_walk(&mut self, n: f32) {
        for i in 0..self.arena.len() {
            self.arena[i].x += n;
        }
    }

    fn apportion(&mut self, index: usize, default_ancestor: usize, distance: f32) -> usize {
        let w = self.left_brother(index);
        let mut default_ancestor = default_ancestor;
        if let Some(w) = w { 
            //in buchheim notation:
            //i == inner; o == outer; r == right; l == left; r = +; l = -
            let mut vir = index;
            let mut vor = index;
            let mut vil = w;
            let mut vol = self.leftmost_sibling(vir).unwrap();
            let mut sir = self.arena[index].module;
            let mut sor = self.arena[index].module;
            let mut sil = self.arena[vil].module;
            let mut sol = self.arena[vol].module;
            println!("vil: {}, vir: {}, vol: {}, vor: {}", vil, vir, vol, vor);
            while self.right(vil) != None && self.left(vir) != None {
                vil = self.right(vil).unwrap();
                vir = self.left(vir).unwrap();
                // TODO: This unwrap may panic, check if makes sence to unwrap
                // Try unwrap_or 0
                vol = self.left(vol).unwrap_or(0);
                vor = self.right(vor).unwrap_or(0);
                self.arena[vor].ancestor = index;
                let shift = (self.arena[vil].x + sil) - (self.arena[vir].x + sir) + distance;
                if shift > 0.0 {
                    self.move_subtree(self.ancestor(vil, index, default_ancestor), index, shift);
                    sir += shift;
                    sor += shift;
                }
                sil += self.arena[vil].module;
                sir += self.arena[vir].module;
                sol += self.arena[vol].module;
                sor += self.arena[vor].module;
            }

            if self.right(vil) != None && self.right(vol) == None {
                self.arena[vor].thread = self.right(vil);
                self.arena[vor].module += sil - sor;
            } 
            if self.left(vir) != None && self.left(vol) == None {
                self.arena[vol].thread = self.left(vir);
                self.arena[vol].module += sir - sol;
                default_ancestor = index;
            }
        }
        return default_ancestor;
    }

    fn move_subtree(&mut self, wl: usize, wr: usize, shift: f32) {
        let subtrees = self.arena[wr].number - self.arena[wl].number;
        println!("{}", subtrees);
        self.arena[wr].change -= shift / subtrees as f32;
        self.arena[wl].change += shift / subtrees as f32;
        self.arena[wr].shift += shift;
        self.arena[wr].x += shift;
        self.arena[wr].module += shift;
    }

    fn ancestor(&self, vil: usize, v: usize, default_ancestor: usize) -> usize {
        // The relevant text is at the bottom of page 7 of
        // "Improving Walker's Algorithm to Run in Linear Time" by Buchheim et al, (2002)
        // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8757&rep=rep1&type=pdf
        if self.arena[self.arena[v].parent.unwrap()].children.contains(&self.arena[vil].ancestor) {
            return self.arena[vil].ancestor;
        }
        return default_ancestor; 
    }

    fn execute_shifts(&mut self, index: usize) {
        let mut shift = 0.0;
        let mut change = 0.0;
        for &child in self.arena[index].children.clone().iter().rev() {
            self.arena[child].x += shift;
            self.arena[child].module += shift;
            change += self.arena[child].change;
            shift += self.arena[child].shift + change;
        }
    }

    pub fn root_path(&self, index: usize) -> Vec<T> {
        let mut path: Vec<T> = vec![];
        let mut current = self.get_node(index);
        while let Some(c) = current {
            path.push(c.val.clone());
            if let Some(parent) = c.parent {
                current = self.get_node(parent);
            } else {
                current = None;
            }
        }
        path
    }

    pub fn left_brother(&self, index: usize) -> Option<usize> {
        if let Some(node) = self.get_node(index) {
            if let Some(parent_index) = node.parent {
                let parent = self.get_node(parent_index).expect("Node with invalid parent.");
                let mut n: Option<usize> = None;
                for &c in &parent.children {
                    if index == c {
                        return n;
                    }
                    n = Some(c);
                }
            }
        }
        None
    }

    pub fn leftmost_sibling(&self, index: usize) -> Option<usize> {
        if let Some(parent) = self.get_parent(index) {
            if parent.children[0] != index {
                return Some(parent.children[0])
            }
        }
        None
    }

    pub fn get_parent(&self, index: usize) -> Option<&Node<T>> {
        if let Some(node) = self.get_node(index) {
            if let Some(parent_index) = node.parent {
                return self.get_node(parent_index);
            }
        }
        None
    }

    pub fn iterative_postorder(&self) -> Vec<NodeVisualizer<T>> {
        let mut stack: Vec<NodeVisualizer<T>> = Vec::new();
        let mut res: Vec<NodeVisualizer<T>> = Vec::new();
        let root = self.get_node(0);
        if let Some(r) = root {
            stack.push(NodeVisualizer { node: r, x: 0 });
        }
        while !stack.is_empty() {
            let visualizer = stack.pop().unwrap();
            res.push(visualizer.clone());
            for (i, child_index) in visualizer.node.children.iter().enumerate() {
                let child = self.get_node(*child_index).unwrap();
                stack.push(NodeVisualizer { node: child, x: i });
            }
        
        }
        let rev_iter = res.iter().rev();
        let mut rev: Vec<NodeVisualizer<T>> = Vec::new();
        for elem in rev_iter {
            rev.push(elem.clone());
        }
        rev
    }
}

pub struct Solver<'a> {
    initial_board: Board,
    goal: Board,
    heuristic: &'a dyn Fn(&Board, &Board) -> usize,
    game_tree: ArenaTree<Board>,
    frontier: Vec<State>,
    result_path: Vec<Board>,
    solved: bool,
}

impl Solver<'_> {
    pub fn new(initial_board: Board, goal: Board, heuristic: &dyn Fn(&Board, &Board) -> usize) -> Solver {
        let mut game_tree: ArenaTree<Board> = ArenaTree::new();
        game_tree.insert(initial_board, None);

        let mut frontier = vec![];
        let initial_state = State::new(initial_board, 0, 0, None);
        frontier.push(initial_state);

        Solver { initial_board, goal, heuristic, game_tree, frontier, solved: false, result_path: vec![] }
    }

    pub fn step(&mut self) {
        if !self.solved && !self.frontier.is_empty() {
            if let Some(&current) = self.frontier.iter().min_by(|x, y| x.cost.cmp(&y.cost)) {
                if current.board == self.goal {
                    self.solved = true;
                    if let Some(current_index) = self.game_tree.get_index(&current.board) {
                        self.result_path = self.game_tree.root_path(current_index);
                    }
                    return;
                }
                self.frontier.retain(|&b| b != current);
                let parent_index = self.game_tree.get_index(&current.board);
                let neighbors = current.next_states(&self.goal, self.heuristic);
                for n in neighbors {
                    self.game_tree.insert(n.board, parent_index);
                    self.frontier.push(n);
                }
            }
        }
    }

    pub fn is_solved(&self) -> bool {
        self.solved
    }

    pub fn result_path(&self) -> &Vec<Board> {
        &self.result_path
    }

    pub fn game_tree(&mut self) -> &mut ArenaTree<Board> {
        &mut self.game_tree
    }
}

pub fn resolve(initial_board: Board, goal: Board, heuristic: &dyn Fn(&Board, &Board) -> usize) -> Vec<Board> {
    let mut solver = Solver::new(initial_board, goal, &heuristic);
    while !solver.is_solved() {
        solver.step();
    }
    solver.result_path().clone()
}

pub fn hamming(goal: &Board, board: &Board) -> usize {
    let mut displaced = 0;
    for i in 0..3 {
        for j in 0..3 {
            if goal.0[i][j] != board.0[i][j] {
                displaced += 1;
            }
        }
    }
    displaced
}
