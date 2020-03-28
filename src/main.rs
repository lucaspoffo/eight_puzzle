extern crate imgui;

use eight_puzzle::{hamming, Board, Solver, Node};
use imgui::*;

mod support;


fn main() {
    // let system = support::init(file!());
    // system.main_loop(move |run, ui| ui.show_demo_window(run));
    
    let system = support::init(file!());
    // let initial_board: Board = Board::new([[2, 3, 6], [1, 5, 8], [4, 0, 7]]);
    let initial_board: Board = Board::new([[1, 3, 4], [8, 0, 5], [7, 2, 6]]);
    // let initial_board: Board = Board::new([[1, 6, 2], [5, 7, 3], [0, 4, 8]]);
    let goal: Board = Board::new([[1, 2, 3], [4, 5, 6] ,[7, 8, 0]]);

    let mut solver = Solver::new(initial_board, goal, &hamming);
    let mut count = 0;
    system.main_loop(move |_, ui| {
        let window = Window::new(im_str!("ImGui Demo"))
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .scroll_bar(true)
            .collapsible(false)
            .menu_bar(false)
            .size( ui.io().display_size, Condition::Always)
            .position([0.0, 0.0], Condition::FirstUseEver);
        window.build(ui, || {
            if ui.button(im_str!("Step"), [100.0, 20.0]) {
                if !solver.is_solved() {
                    count += 1;
                }
                println!("-------------------------");
                solver.step();
                solver.game_tree_mut().adjust_visualization();
            }
            for i in 0..solver.game_tree().arena.len() {
                let node = &solver.game_tree().arena[i];
                let pos = [200.0 + 100.0 * node.x, 100.0 * node.depth as f32];
                if let Some(parent) = node.parent {
                    let draw_list = ui.get_window_draw_list();
                    let parent_node = &solver.game_tree().arena[parent];
                    let parent_pos = [200.0 + 100.0 * parent_node.x - ui.scroll_x(), 100.0 * parent_node.depth as f32 - ui.scroll_y()];
                    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
                    draw_list.add_line([pos[0] - ui.scroll_x(), pos[1] - ui.scroll_y()], parent_pos, WHITE).build();
                }
                ui.set_cursor_pos(pos);
                ui.text(im_str!("{}x: {}\ny: {}\nm: {}\n",  node.val.to_string(), node.x, node.depth, node.modifier));
            }
        })
    });
}
