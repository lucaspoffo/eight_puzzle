use eight_puzzle::{resolve, hamming, Board};


fn main() {
    let initial_board: Board = Board::new([[1, 6, 2], [5, 7, 3], [0, 4, 8]]);
    let goal: Board = Board::new([[1, 2, 3], [4, 5, 6] ,[7, 8, 0]]);

    let moves = resolve(initial_board, &goal, &hamming);
    for m in moves {
        println!("{}", m.board);
        println!("-----");
    }
}
