mod boundary;
mod control_regions;
mod layering;
mod partition;
mod render;
mod wires;

use std::path::PathBuf;

use crate::compile::{CompileGraph, CompileTheory};

use self::{
    boundary::BoundaryWires,
    control_regions::process_control_regions,
    partition::{partition_control_regions, partition_data_regions},
    render::{graph_svg, render_graph_region_svgs, render_region_svgs},
    wires::assert_interleaved_control_operations_are_unary,
};

pub use control_regions::ControlRegionGraph;
pub use layering::NestedGraph;

pub fn render_analysis(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    Ok(render_analysis_artifacts(graph)?
        .into_iter()
        .find(|artifact| artifact.path == PathBuf::from("normalized.svg"))
        .expect("analysis artifacts include normalized graph")
        .contents)
}

#[derive(Debug, Clone)]
pub struct AnalysisArtifact {
    pub path: PathBuf,
    pub contents: Vec<u8>,
}

pub fn control_region_graphs(graph: &CompileGraph) -> Vec<ControlRegionGraph> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );
    assert_interleaved_control_operations_are_unary(&graph.graph);
    let regions = partition_data_regions(&graph.graph);
    partition_control_region_graphs(process_control_regions(graph, &regions))
}

pub fn render_analysis_artifacts(graph: &CompileGraph) -> std::io::Result<Vec<AnalysisArtifact>> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );

    // I don't know if it is too strict, but I cannot imagine a case when it is not true
    // better fail early and loud if I am wrong!
    assert_interleaved_control_operations_are_unary(&graph.graph);
    let _boundary_wires = BoundaryWires::from_graph(&graph.graph);
    let regions = partition_data_regions(&graph.graph);
    let region_svgs = render_region_svgs(graph, &regions)?;
    let control_region_graphs =
        partition_control_region_graphs(process_control_regions(graph, &regions));
    let before_split = graph_svg(&graph.graph)?;
    let mut artifacts = vec![
        AnalysisArtifact {
            path: PathBuf::from("normalized.svg"),
            contents: before_split.clone(),
        },
        AnalysisArtifact {
            path: PathBuf::from("before-split.svg"),
            contents: before_split.clone(),
        },
    ];
    artifacts.extend(region_svgs.into_iter().map(|region| AnalysisArtifact {
        path: PathBuf::from("regions").join(region.file_name),
        contents: region.svg,
    }));
    for control_region in control_region_graphs {
        artifacts.push(AnalysisArtifact {
            path: PathBuf::from("control-regions")
                .join(format!("{:03}-resolved.svg", control_region.region_index)),
            contents: graph_svg(&control_region.nested_graph.graph)?,
        });
        artifacts.extend(
            render_graph_region_svgs(&control_region.nested_graph.graph, &control_region.regions)?
                .into_iter()
                .map(|region| AnalysisArtifact {
                    path: PathBuf::from("control-regions")
                        .join(format!("{:03}-regions", control_region.region_index))
                        .join(region.file_name),
                    contents: region.svg,
                }),
        );
    }
    Ok(artifacts)
}

fn partition_control_region_graphs(
    control_region_graphs: Vec<ControlRegionGraph>,
) -> Vec<ControlRegionGraph> {
    control_region_graphs
        .into_iter()
        .map(|mut control_region| {
            control_region.regions = partition_control_regions(&control_region.nested_graph.graph);
            control_region
        })
        .collect()
}
