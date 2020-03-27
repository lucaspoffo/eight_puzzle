extern crate imgui;

use eight_puzzle::{hamming, Board, Solver, Node};
use imgui::*;

mod support;


fn main() {
    // let system = support::init(file!());
    // system.main_loop(move |run, ui| ui.show_demo_window(run));
    
    let system = support::init(file!());
    let initial_board: Board = Board::new([[1, 6, 2], [5, 7, 3], [0, 4, 8]]);
    let goal: Board = Board::new([[1, 2, 3], [4, 5, 6] ,[7, 8, 0]]);

    let mut solver = Solver::new(initial_board, goal, &hamming);
    let mut count = 0;
    let mut once = true;
    system.main_loop(move |_, ui| {
        solver.step();
        if !solver.is_solved() {
            count += 1;
        } else if once {
            solver.game_tree().buchheim();
            once = false;
        }
        for node in &solver.game_tree().arena {
            // Window::new(im_str!("Hello world"))
            //     .size([300.0, 300.0], Condition::FirstUseEver)
            //     .position([node.x * (400.0 as f32), (node.depth as f32 * 400.0) as f32], Condition::Always)
            //     .build(ui, || {
            //         ui.text(im_str!("{}x: {}\ny: {}\n",  node.val.to_string(), node.x, node.depth));
            //     });
        }
        Window::new(im_str!("Hello world"))
        .size([300.0, 300.0], Condition::FirstUseEver)
        .build(ui, || {
            
            // let current = solver.game_tree().get_node(0);
            // while let Some(c) = current {
            //     c.children
            // }
            // let tree: Vec<String> = solver.game_tree().iterative_postorder().iter().map(|v| {
            //     format!("{}x: {}\ny: {}\n", v.node.val.to_string(), v.x, v.node.depth)
            // }).collect();
            
            // println!("{:?}", solver.game_tree());
            ui.text(im_str!("Is done: {}", solver.is_solved()));
            ui.text(im_str!("Steps: {}", count));
            for node in &solver.game_tree().arena {
                ui.text(im_str!("{}x: {}\ny: {}\n {}\n",  node.val.to_string(), node.x, node.depth, node.number));
            }
            ui.separator();
            let mouse_pos = ui.io().mouse_pos;
            ui.text(format!(
                "Mouse Position: ({:.1},{:.1})",
                mouse_pos[0], mouse_pos[1]
            ));
        });
    });

    // while !solver.is_solved() {
    //     solver.step();
    // }

    // for b in solver.result_path() {
    //     println!("{}", b);
    //     println!("-----");
    // }
}
