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
        write!(f, "{} {} {}\n{} {} {}\n{} {} {}", 
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
struct ArenaTree<T> where T: PartialEq + Clone {
    arena: Vec<Node<T>>
}

#[derive(Debug, Clone)]
struct Node<T> where T: PartialEq + Clone {
    index: usize,
    val: T,
    parent: Option<usize>,
    children: Vec<usize>
}

impl<T> Node<T> where T: PartialEq + Clone {
    fn new(index: usize, val: T, parent: Option<usize>) -> Self {
        Self { index, val, parent, children: vec![] }
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

    fn get_node(&self, index: usize) -> Option<&Node<T>> {
        if index < self.arena.len() {
            return Some(&self.arena[index]);
        }
        None
    }
    
    fn insert(&mut self, val: T, parent_index: Option<usize>) {
        let index = self.arena.len();
        self.arena.push(Node::new(index, val, parent_index));
        if let Some(parent_index) = parent_index {
            self.arena[parent_index].children.push(index);
        }
    }

    fn root_path(&self, index: usize) -> Vec<T> {
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
