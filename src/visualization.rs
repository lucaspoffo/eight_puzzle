extern crate imgui;

use super::{hamming, manhattan, Board, Solver, Node, a_star, State, breadth_first, HeuristicFn, PathFn};
use imgui::*;

static SIDE_BAR_WIDTH: f32 = 200.0;
static GRAPH_SPACING: f32 = 50.0;
static NODE_DISTANCE_X: f32 = 100.0;
static NODE_DISTANCE_Y: f32 = 100.0;

static NODE_WIDTH: f32 = 55.0;
static NODE_HEIGHT: f32 = 50.0;

static WHITE: [f32; 3] = [1.0, 1.0, 1.0];
static RED: [f32; 3] = [1.0, 0.0, 0.0];

pub struct VisualizationState<'a> {
  solver: Solver<'a>,
  cost_func: HeuristicFn<'a>,
  path_algorithm: PathFn<'a>,
  board_input: [[i32; 3]; 3],
  goal_input: [[i32; 3]; 3],
  initial_board: Board,
  goal: Board,
  steps: usize,
  is_solving: bool,
  update_visualization: bool,
  adjust_once: bool,
}

impl Default for VisualizationState<'_> {
	fn default() -> Self {
		let initial_board: Board = Board::new([[1, 6, 2], [5, 7, 3], [0, 4, 8]]);
		let board_input: [[i32; 3]; 3] = [[1, 6, 2], [5, 7, 3], [0, 4, 8]];
		let goal: Board = Board::new([[1, 2, 3], [4, 5, 6] ,[7, 8, 0]]);
		let goal_input: [[i32; 3]; 3] = [[1, 2, 3], [4, 5, 6] ,[7, 8, 0]];
		
		let solver = Solver::new(initial_board, goal, &hamming, &a_star);

		VisualizationState {
			solver,
			cost_func: &hamming,
			board_input,
			goal_input,
			initial_board,
			goal,
			steps: 0,
			is_solving: false,
			update_visualization: true,
			adjust_once: true,
			path_algorithm: &a_star
		}
	}
}

impl VisualizationState<'_> {
	fn reset(&mut self) {
		self.steps = 0;
		self.is_solving = false;
		self.adjust_once = true;
		self.solver = Solver::new(self.initial_board, self.goal, self.cost_func, self.path_algorithm);
	}
}

pub fn show_visualization_window(ui: &Ui, state: &mut VisualizationState) {
	let solver = &mut state.solver;
	if state.is_solving {
		if !solver.is_solved() {
			state.steps += 1;
		}
		solver.step();
		if state.update_visualization {
			solver.game_tree_mut().adjust_visualization();
		}
	}
	if solver.is_solved() && state.adjust_once {
		solver.game_tree_mut().adjust_visualization();
		state.adjust_once = false;
	}
	show_sidebar(ui, state);
	show_tree_visualizer(ui, state);
}

fn show_sidebar(ui: &Ui, state: &mut VisualizationState) {
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
		ui.text(im_str!("Initial board:"));
		ui.input_int3(im_str!("b1"), &mut state.board_input[0]).build();
		ui.input_int3(im_str!("b2"), &mut state.board_input[1]).build();
		ui.input_int3(im_str!("b3"), &mut state.board_input[2]).build();
		let board = Board::from(state.board_input);
		if !board.is_valid() {
			ui.text_colored([1.0, 0.0, 0.0, 1.0], im_str!("Invalid board!"));
		} else if state.initial_board != board {
				state.initial_board = board;
				state.reset();
		}
		ui.separator();
		ui.text(im_str!("Goal board:"));
		ui.input_int3(im_str!("g1"), &mut state.goal_input[0]).build();
		ui.input_int3(im_str!("g2"), &mut state.goal_input[1]).build();
		ui.input_int3(im_str!("g3"), &mut state.goal_input[2]).build();
		let goal = Board::from(state.goal_input);
		if !goal.is_valid() {
			ui.text_colored([1.0, 0.0, 0.0, 1.0], im_str!("Invalid goal!"));
		} else if state.goal != goal {
				state.goal = goal;
				state.reset();
		}

		path_algorithm_selection(ui, state);
		ui.separator();
		ui.checkbox(im_str!("Step Automatically"), &mut state.is_solving);
		ui.checkbox(im_str!("Update visualization"), &mut state.update_visualization);
		ui.separator();
		ui.text(im_str!("Steps: {}", state.steps));
		if ui.button(im_str!("Step"), [100.0, 20.0]) {
			if !state.solver.is_solved() {
				state.steps += 1;
			}
			state.solver.step();
			state.solver.game_tree_mut().adjust_visualization();
		}
		if ui.button(im_str!("Reset"), [100.0, 20.0]) {
			state.reset();
		}
	});

}

fn show_tree_visualizer(ui: &Ui, state: &mut VisualizationState) {
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
		if !state.update_visualization && !state.solver.is_solved() { return; }
		for i in 0..state.solver.game_tree().arena.len() {
			let node = &state.solver.game_tree().arena[i];
			let mut parent_node = None;
			if let Some(parent) = node.parent {
				parent_node = Some(&state.solver.game_tree().arena[parent]);
			}
			show_node(ui, node, parent_node);
		}
	})
}

fn show_node(ui: &Ui, node: &Node<State>, parent: Option<&Node<State>>) {
	let window_pos = ui.window_pos();
	let pos = calculate_node_pos(node.x, node.depth);
	let draw_offset = [window_pos[0] - ui.scroll_x(), window_pos[1] - ui.scroll_y()];
	let draw_pos = [pos[0] + draw_offset[0], pos[1] + draw_offset[1]];
	let draw_list = ui.get_window_draw_list();
	let rect_start_x = draw_pos[0]; 
	let rect_stary_y = draw_pos[1];
	let rect_end_x = draw_pos[0] + NODE_WIDTH;
	let rect_end_y =  draw_pos[1] + NODE_HEIGHT;
	let color = if node.is_solution { RED } else { WHITE };
	draw_list.add_rect([rect_start_x, rect_stary_y], [rect_end_x, rect_end_y], color).build();
	if let Some(parent) = parent {
		let parent_pos = calculate_node_pos(parent.x, parent.depth);
		let parent_middle_bottom = [parent_pos[0] + NODE_WIDTH / 2.0 + draw_offset[0], parent_pos[1] + NODE_HEIGHT + draw_offset[1]];
		let node_middle_top = [draw_pos[0] + NODE_WIDTH / 2.0, draw_pos[1]];
		draw_list.add_line(parent_middle_bottom, node_middle_top, color).build();
	}
	ui.set_cursor_pos([pos[0] + 10.0, pos[1] + 5.0]);
	ui.text(im_str!("{}",  node.val.board.to_string()));
}

fn calculate_node_pos(x: f32, y: usize) -> [f32; 2] {
    [GRAPH_SPACING + NODE_DISTANCE_X * x, GRAPH_SPACING + NODE_DISTANCE_Y * y as f32]
}

fn path_algorithm_selection(ui: &Ui, state: &mut VisualizationState) {
	ui.separator();
	ui.text(im_str!("Algorithm:"));
	if option_select(ui, state.path_algorithm, &a_star, im_str!("A*")) {
		state.path_algorithm = &a_star;
		state.reset();
	}
	if option_select(ui, state.path_algorithm, &super::greedy_best, im_str!("Greedy Best")) {
		state.path_algorithm = &super::greedy_best;
		state.reset();
	}
	if option_select(ui, state.path_algorithm, &breadth_first, im_str!("Breadth First")) {
		state.path_algorithm = &breadth_first;
		state.reset();
	}
	if option_select(ui, state.path_algorithm, &super::depth_first, im_str!("Depth First")) {
		state.path_algorithm = &super::depth_first;
		state.reset();
	}

	// TODO: investigate why calling ptr::eq from show_sidebar does not return the same value from here.
	if std::ptr::eq(state.path_algorithm, &a_star) || std::ptr::eq(state.path_algorithm, &super::greedy_best) {
		cost_function_selection(ui, state);
	}
}

fn cost_function_selection(ui: &Ui, state: &mut VisualizationState) {
	ui.separator();
	ui.text(im_str!("Cost Function:"));
	if option_select(ui, state.cost_func, &hamming, im_str!("Hamming")) {
		state.cost_func = &hamming;
		state.reset();
	}
	if option_select(ui, state.cost_func, &manhattan, im_str!("Manhattan")) {
		state.cost_func = &manhattan;
		state.reset();
	}
}

fn option_select<T: ?Sized>(ui: &Ui, selection: &T, option: &T, label: &ImStr) -> bool {
	let is_selected = std::ptr::eq(selection, option);
	let select = Selectable::new(label)
		.selected(is_selected)
		.build(ui);
	select && !is_selected
}
