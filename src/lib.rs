use std::collections::VecDeque;
use std::cmp::{Eq, PartialEq};
use std::fmt;
use std::f32;

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
    pub modifier: f32,
}

impl<T> Node<T> where T: PartialEq + Clone {
    fn new(index: usize, val: T, parent: Option<usize>, depth: usize) -> Self {
        Self { index, val, parent, children: vec![], depth, x: 0.0, thread: None, modifier: 0.0 }
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

    pub fn calculate_inital_x(&mut self) {
        let mut stack: Vec<usize> = Vec::new();
        if self.arena.len() > 0 {
            stack.push(0);
        }
        
        while !stack.is_empty() {
            let index = stack.pop().unwrap();
            for i in 0..self.arena[index].children.len() {
                let child_index = self.arena[index].children[i];
                stack.push(child_index);
                self.arena[child_index].x = i as f32;
            }
        }
    }

    pub fn adjust_visualization(&mut self) {
        for i in 0..self.arena.len() {
            self.arena[i].modifier = 0.0;
            self.arena[i].x = 0.0;
        }
        // self.calculate_inital_x();
        self.centralize_parent();
        self.calculate_final_x();
        self.fix_conflicts();
        // self.calculate_final_x();
        // self.fix_parent();
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
                        let child_index = self.arena[index].children[0];
                        // self.arena[index].x = self.arena[child_index].x;
                        // self.arena[index].modifier = self.arena[child_index].x;
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
                    // for i in 0..self.arena[index].children.len() - 1 {
                    //     let left_child = self.arena[index].children[i];
                    //     let right_child = self.arena[index].children[i + 1];
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

    fn fix_parent(&mut self) {
        let mut nodes = vec![0];
        while !nodes.is_empty() {
            let node = nodes.pop().unwrap();
            let len = self.arena[node].children.len();
            if len > 1 {
                let left_child = self.arena[node].children[0];
                let right_child = self.arena[node].children[len - 1];
                self.arena[node].x = (self.arena[left_child].x + self.arena[right_child].x) / 2.0;
            }
            for i in 0..self.arena[node].children.len() {
                let child_index = self.arena[node].children[i];
                nodes.push(child_index);
            }
        }
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

    pub fn game_tree_mut(&mut self) -> &mut ArenaTree<Board> {
        &mut self.game_tree
    }

    pub fn game_tree(&self) -> &ArenaTree<Board> {
        &self.game_tree
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
