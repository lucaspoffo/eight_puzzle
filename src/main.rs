extern crate imgui;

use eight_puzzle::{hamming, Board, Solver, visualization};
use imgui::*;

mod support;

fn main() {
    // let system = support::init(file!());
    // system.main_loop(move |run, ui| ui.show_demo_window(run));

    let mut state = visualization::State::default();

    let system = support::init(file!());

    system.main_loop(move |_, ui| {
        visualization::show_visualization_window(&ui, &mut state);
    });
}
