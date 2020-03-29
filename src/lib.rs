use std::cmp::{Eq, PartialEq};
use std::fmt;
use std::f32;

pub use self::visualization::*;
pub mod visualization;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Board(pub [[usize; 3]; 3]);

impl Board {
  pub fn new(board: [[usize; 3]; 3]) -> Board {
    Board(board)
  }

  pub fn is_valid(&self) -> bool {
    // Has all correct board values
    'outer: for value in 0..9 {
        for i in 0..3 {
            for j in 0..3 {
                if value == self.0[i][j] {
                    continue 'outer;
                }
            }            
        }
        return false;
    }

    // Validate number of inversions, if is even it's solvable
    // https://www.geeksforgeeks.org/check-instance-8-puzzle-solvable/
    let mut inversion_count = 0;
    for i in 0..3 {
        for j in 0..3 {
            for k in (i * 3 + j + 1)..9 {
                let x = k / 3;
                let y = k % 3;
                if self.0[i][j] != 0 && self.0[x][y] != 0 && self.0[i][j] > self.0[x][y] {
                    inversion_count += 1;
                }    
            }
        }
    }
    inversion_count % 2 == 0 
  }
}

impl From<[[i32; 3]; 3]> for Board {
    fn from(x: [[i32; 3]; 3]) -> Self {
        let mut r = [[0,0,0], [0,0,0], [0,0,0]];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = x[i][j] as usize;
            }
        }
        Board(r)
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
           let cost = heuristic(&goal, &board);
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
    pub modifier: f32,
    pub is_solution: bool,
}

impl<T> Node<T> where T: PartialEq + Clone {
    fn new(index: usize, val: T, parent: Option<usize>, depth: usize) -> Self {
        Self { index, val, parent, children: vec![], depth, x: 0.0, thread: None, modifier: 0.0, is_solution: false }
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
        if let Some(index) = parent_index {
            let parent_node = self.get_node(index).expect("Cannot insert node with invalid parent");
            depth = parent_node.depth + 1;
        }
        self.arena.push(Node::new(index, val, parent_index, depth));
        if let Some(parent_index) = parent_index {
            self.arena[parent_index].children.push(index);
        }
    }

    pub fn adjust_visualization(&mut self) {
        for i in 0..self.arena.len() {
            self.arena[i].modifier = 0.0;
            self.arena[i].x = 0.0;
        }
        self.centralize_parent();
        self.calculate_final_x();
        self.fix_conflicts();
        self.shift_min_x();
    }

    pub fn centralize_parent(&mut self) {
        struct Snapshot {
            index: usize,
            stage: usize,
        }
        let mut stack: Vec<Snapshot> = Vec::new();
        if self.arena.len() > 0 {
            stack.push(Snapshot { index: 0, stage: 0 });
        }
        
        while !stack.is_empty() {
            let Snapshot { index, stage } = stack.pop().unwrap();
            match stage {
                0 => {                    
                    stack.push(Snapshot { index, stage: 1 });
                    for i in 0..self.arena[index].children.len() {
                        let child_index = self.arena[index].children[i];
                        self.arena[child_index].x = i as f32;
                        stack.push(Snapshot { index: child_index, stage: 0 });
                    }
                },
                1 => {
                    let children_len = self.arena[index].children.len();
                    if  children_len == 1 {
                        self.arena[index].modifier = self.arena[index].x;
                    } else if children_len > 1 {
                        let left_child = self.arena[index].children[0];
                        let right_child = self.arena[index].children[children_len -1];
                        let desired_x = (self.arena[left_child].x + self.arena[right_child].x) / 2.0;
                        self.arena[index].modifier = self.arena[index].x - desired_x;
                    }
                },
                _ => ()
            }
        }
    }

    pub fn calculate_final_x(&mut self) {
        struct Snapshot {
            index: usize,
            modifier_sum: f32,
        }
        let mut stack: Vec<Snapshot> = Vec::new();
        if self.arena.len() > 0 {
            stack.push(Snapshot { index: 0, modifier_sum: 0.0 });
        }
        
        while !stack.is_empty() {
            let Snapshot { index, modifier_sum } = stack.pop().unwrap();
            self.arena[index].x += modifier_sum;
            for i in 0..self.arena[index].children.len() {
                let child_index = self.arena[index].children[i];
                stack.push(Snapshot { index: child_index, modifier_sum: modifier_sum + self.arena[index].modifier });
            }
        }
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

    fn fix_conflicts(&mut self) {
        struct Snapshot {
            index: usize,
            stage: usize,
        }
        let mut stack: Vec<Snapshot> = Vec::new();
        if self.arena.len() > 0 {
            stack.push(Snapshot { index: 0, stage: 0 });
        }
        
        while !stack.is_empty() {
            let Snapshot { index, stage } = stack.pop().unwrap();
            match stage {
                0 => {
                    let len = self.arena[index].children.len();
                    if len > 0 {
                        stack.push(Snapshot { index, stage: 1 });
                        for i in (0..len).rev() {
                            let child_index = self.arena[index].children[i];
                            stack.push(Snapshot { index: child_index, stage: 0 });
                        }
                    }
                },
                1 => {
                    for i in 0..self.arena[index].children.len() - 1 {
                        for j in (i + 1)..self.arena[index].children.len() {
                            let left_child = self.arena[index].children[i];
                            let right_child = self.arena[index].children[j];
                            let left_contour = self.left_contour(right_child);
                            let right_contour = self.right_contour(left_child);

                            let min_len = usize::min(left_contour.len(), right_contour.len());
                            let mut max_conflict = f32::NEG_INFINITY;
                            for i in 0..min_len {
                                let conflict = right_contour[i] - left_contour[i];
                                max_conflict = f32::max(max_conflict, conflict); 
                            }
                            max_conflict += 1.0;
                            
                            if max_conflict > 0.0 {
                                self.shift(right_child, max_conflict);
                            }
                        }
                    }
                },
                _ => ()
            }
        }
    }

    fn shift_min_x(&mut self) {
        let left_contour = self.left_contour(0);
        let mut min = f32::INFINITY;
        for i in left_contour {
            min = f32::min(min, i);
        }
        self.shift(0, -min);
    }

    fn left_contour(&self, root: usize) -> Vec<f32> {
        let mut nodes = vec![root];
        let mut contour = vec![];
        contour.push(f32::INFINITY);
        let root_depth = self.arena[root].depth;
        while !nodes.is_empty() {
            let node = nodes.pop().unwrap();
            let depth = self.arena[node].depth - root_depth;
            if contour.len() - 1 < depth {
                contour.push(self.arena[node].x);
            } else {
                contour[depth] = f32::min(contour[depth], self.arena[node].x);
            }
            for &child in &self.arena[node].children {
                nodes.push(child);
            }
        }
        contour
    }

    fn right_contour(&self, root: usize) -> Vec<f32> {
        let mut nodes = vec![root];
        let mut contour = vec![];
        let root_depth = self.arena[root].depth;
        contour.push(f32::NEG_INFINITY);
        while !nodes.is_empty() {
            let node = nodes.pop().unwrap();
            let depth = self.arena[node].depth - root_depth;
            if contour.len() - 1 < depth {
                contour.push(self.arena[node].x);
            } else {
                contour[depth] = f32::max(contour[depth], self.arena[node].x);
            }
            for &child in &self.arena[node].children {
                nodes.push(child);
            }
        }
        contour
    }

    fn shift(&mut self, root: usize, value: f32) {
        let mut nodes = vec![root];
        while !nodes.is_empty() {
            let node = nodes.pop().unwrap();
            self.arena[node].x += value;
            for i in 0..self.arena[node].children.len() {
                let child_index = self.arena[node].children[i];
                nodes.push(child_index);
            }
        }
    }

    pub fn solution_path(&mut self, index: usize) -> Vec<T> {
        let mut path: Vec<T> = vec![];
        let mut current = self.get_node_mut(index);
        while let Some(c) = current {
            c.is_solution = true;
            path.push(c.val.clone());
            if let Some(parent) = c.parent {
                current = self.get_node_mut(parent);
            } else {
                current = None;
            }
        }
        path
    }
}

pub type HeuristicFn<'a> = &'a dyn Fn(&Board, &Board) -> usize;
pub type PathFn<'a> = &'a dyn Fn(&Vec<State>) -> Option<&State>;

pub struct Solver<'a> {
    goal: Board,
    heuristic: HeuristicFn<'a>,
    path_algorithm: PathFn<'a>,
    game_tree: ArenaTree<State>,
    frontier: Vec<State>,
    result_path: Vec<State>,
    solved: bool,
}

impl<'a> Solver<'a> {
    pub fn new(initial_board: Board, goal: Board, heuristic: HeuristicFn<'a>, path_algorithm: PathFn<'a>) -> Solver<'a> {
        let mut game_tree: ArenaTree<State> = ArenaTree::new();
        
        let mut frontier = vec![];
        let initial_state = State::new(initial_board, 0, 0, None);
        game_tree.insert(initial_state, None);
        frontier.push(initial_state);

        Solver { goal, heuristic, game_tree, frontier, solved: false, result_path: vec![], path_algorithm }
    }

    pub fn step(&mut self) {
        if !self.solved && !self.frontier.is_empty() {
            if let Some(&current) = (self.path_algorithm)(&self.frontier) {
                if current.board == self.goal {
                    self.solved = true;
                    if let Some(current_index) = self.game_tree.get_index(&current) {
                        self.result_path = self.game_tree.solution_path(current_index);
                    }
                    return;
                }
                self.frontier.retain(|&b| b != current);
                let parent_index = self.game_tree.get_index(&current);
                let neighbors = current.next_states(&self.goal, self.heuristic);
                for n in neighbors {
                    self.game_tree.insert(n, parent_index);
                    self.frontier.push(n);
                }
            }
        }
    }

    pub fn is_solved(&self) -> bool {
        self.solved
    }

    pub fn result_path(&self) -> &Vec<State> {
        &self.result_path
    }

    pub fn game_tree_mut(&mut self) -> &mut ArenaTree<State> {
        &mut self.game_tree
    }

    pub fn game_tree(&self) -> &ArenaTree<State> {
        &self.game_tree
    }
}

pub fn a_star<'a>(frontier: &'a Vec<State>) -> Option<&'a State> {
    frontier.iter().min_by(|x, y| (x.cost + x.moves).cmp(&(y.cost + y.moves)))
}

pub fn greedy_best<'a>(frontier: &'a Vec<State>) -> Option<&'a State> {
    frontier.iter().min_by(|x, y| x.cost.cmp(&y.cost))
}

pub fn breadth_first<'a>(frontier: &'a Vec<State>) -> Option<&'a State> {
    if frontier.len() > 0 {
        return Some(&frontier[0])
    }
    None
}

pub fn depth_first<'a>(frontier: &'a Vec<State>) -> Option<&'a State> {
    if frontier.len() > 0 {
        return Some(&frontier[frontier.len() - 1])
    }
    None
}

pub fn resolve(initial_board: Board, goal: Board, heuristic: &dyn Fn(&Board, &Board) -> usize) -> Vec<State> {
    let mut solver = Solver::new(initial_board, goal, &heuristic, &a_star);
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

pub fn manhattan(goal: &Board, board: &Board) -> usize {
    let mut displaced = 0;
    for i in 0..3 {
        for j in 0..3 {
            let v = board.0[i][j];
            for x in 0..3 {
                for y in 0..3 {
                    if v == goal.0[x][y] {
                        displaced += i.checked_sub(x).unwrap_or_else(|| { x - i });
                        displaced += j.checked_sub(y).unwrap_or_else(|| { y - j });
                    }
                }
            }
        }
    }
    displaced
}

