use eight_puzzle::{hamming, Board, Solver};


fn main() {
    let initial_board: Board = Board::new([[1, 6, 2], [5, 7, 3], [0, 4, 8]]);
    let goal: Board = Board::new([[1, 2, 3], [4, 5, 6] ,[7, 8, 0]]);

    let mut solver = Solver::new(initial_board, goal, &hamming);
    while !solver.is_solved() {
        solver.step();
    }

    for b in solver.result_path() {
        println!("{}", b);
        println!("-----");
    }
}
