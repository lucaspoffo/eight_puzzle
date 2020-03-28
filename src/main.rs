extern crate imgui;

use eight_puzzle::{hamming, Board, Solver, Node};
use imgui::*;

mod support;

static SIDE_BAR_WIDTH: f32 = 200.0;
static GRAPH_SPACING: f32 = 50.0;
static NODE_DISTANCE_X: f32 = 100.0;
static NODE_DISTANCE_Y: f32 = 100.0;

static NODE_WIDTH: f32 = 55.0;
static NODE_HEIGHT: f32 = 50.0;
static ZOOM: f32 = 0.5;

fn main() {
    // let system = support::init(file!());
    // system.main_loop(move |run, ui| ui.show_demo_window(run));
    


    let system = support::init(file!());
    // let initial_board: Board = Board::new([[2, 3, 6], [1, 5, 8], [4, 0, 7]]);
    // let initial_board: Board = Board::new([[1, 3, 4], [8, 0, 5], [7, 2, 6]]);
    let initial_board: Board = Board::new([[1, 6, 2], [5, 7, 3], [0, 4, 8]]);
    let goal: Board = Board::new([[1, 2, 3], [4, 5, 6] ,[7, 8, 0]]);

    let mut solver = Solver::new(initial_board, goal, &hamming);
    let mut count = 0;
    let mut solving = false;
    system.main_loop(move |_, ui| {
        if solving {
            if !solver.is_solved() {
                count += 1;
            }
            solver.step();
            solver.game_tree_mut().adjust_visualization();
        }

        let side_bar = Window::new(im_str!("sidebar"))
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .scroll_bar(true)
            .collapsible(false)
            .menu_bar(false)
            .size([SIDE_BAR_WIDTH, ui.io().display_size[1]], Condition::Always)
            .position([0.0, 0.0], Condition::FirstUseEver);
        side_bar.build(ui, || {
            if ui.button(im_str!("Step"), [100.0, 20.0]) {
                if !solver.is_solved() {
                    count += 1;
                }
                solver.step();
                solver.game_tree_mut().adjust_visualization();
            }
            ui.checkbox(im_str!("Solving"), &mut solving);
        });
        let tree_visualizer = Window::new(im_str!("ImGui Demo"))
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .scroll_bar(true)
            .collapsible(false)
            .menu_bar(false)
            .size([ui.io().display_size[0] - SIDE_BAR_WIDTH, ui.io().display_size[1]], Condition::Always)
            .position([SIDE_BAR_WIDTH, 0.0], Condition::FirstUseEver);
        tree_visualizer.build(ui, || {
            
            for i in 0..solver.game_tree().arena.len() {
                let node = &solver.game_tree().arena[i];
                let window_pos = ui.window_pos();
                let pos = calculate_node_pos(node.x, node.depth);
                let draw_offset = [window_pos[0] - ui.scroll_x(), window_pos[1] - ui.scroll_y()];
                let draw_pos = [pos[0] + draw_offset[0], pos[1] + draw_offset[1]];
                let draw_list = ui.get_window_draw_list();
                let rect_start_x = draw_pos[0]; 
                let rect_stary_y = draw_pos[1];
                let rect_end_x = draw_pos[0] + NODE_WIDTH;
                let rect_end_y =  draw_pos[1] + NODE_HEIGHT;
                draw_list.add_rect([rect_start_x, rect_stary_y], [rect_end_x, rect_end_y], WHITE).build();
                if let Some(parent) = node.parent {
                    let parent_node = &solver.game_tree().arena[parent];
                    let parent_pos = calculate_node_pos(parent_node.x, parent_node.depth);
                    let parent_middle_bottom = [parent_pos[0] + NODE_WIDTH / 2.0 + draw_offset[0], parent_pos[1] + NODE_HEIGHT + draw_offset[1]];
                    let node_middle_top = [draw_pos[0] + NODE_WIDTH / 2.0, draw_pos[1]];
                    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
                    draw_list.add_line(parent_middle_bottom, node_middle_top, WHITE).build();
                }
                const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
                ui.set_cursor_pos([pos[0] + 10.0, pos[1] + 5.0]);
                ui.text(im_str!("{}",  node.val.to_string()));
                // ui.text(im_str!("{}x: {}\ny: {}\nm: {}\n",  node.val.to_string(), node.x, node.depth, node.modifier));
            }
        })
    });
}

fn calculate_node_pos(x: f32, y: usize) -> [f32; 2] {
    return [GRAPH_SPACING + NODE_DISTANCE_X * x, GRAPH_SPACING + NODE_DISTANCE_Y * y as f32];
}
