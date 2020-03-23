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
        board.0[empty_space.0][empty_space.1] = board.0[to.0][to.1];;
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

pub fn resolve(initial_board: Board, goal: &Board, heuristic: &dyn Fn(&Board, &Board) -> usize) -> Vec<State> {
    let inital_state = State::new(initial_board, 0, 0, None);

    let mut open_set: Vec<State> = vec![];
    let mut came_from: HashMap<State, State> = HashMap::new();
    open_set.push(inital_state);

    while !open_set.is_empty() {
        if let Some(&current) = open_set.iter().min_by(|x, y| x.cost.cmp(&y.cost)) {
            if &current.board == goal {
                let mut path = vec![];
                path.push(current);
                let mut from = current;
                while let Some(&x) = came_from.get(&from) {
                    path.push(x);
                    from = x;                    
                }
                return path;
            }
            open_set.retain(|&b| b != current);
            let neighbors = current.next_states(goal, heuristic);
            for n in neighbors {
                came_from.insert(n, current);
                open_set.push(n);
            }
        }
    }
    vec![]
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
