//! Compatibility build pass for Bevy schedule construction.
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::plugin::SubmissionSetConfig;
use bevy_ecs::{
    component::{ComponentDescriptor, ComponentId},
    resource::Resource,
    schedule::{
        InternedSystemSet, NodeId, ScheduleBuildError, ScheduleBuildPass, ScheduleGraph, SystemKey,
        SystemSetKey,
        graph::{Dag, DiGraph},
    },
    system::{IntoSystem, System},
    world::World,
};
use bevy_platform::hash::FixedHasher;
use indexmap::IndexSet;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderSetMetaSystems {
    prelude: SystemKey,
    submission: SystemKey,
    state_id: ComponentId,
}

#[derive(Resource, Default)]
pub(crate) struct SubmissionSetRegistry {
    pub(crate) submission_sets_to_queue:
        HashMap<InternedSystemSet, (ComponentId, SubmissionSetConfig)>,
    /// Maps submission set to its prelude/submission meta-systems.
    pub(crate) submission_sets_to_meta_systems: HashMap<SystemSetKey, RenderSetMetaSystems>,
    /// Maps render sets to its config system
    pub(crate) render_sets_to_systems: HashMap<InternedSystemSet, SystemKey>,
    /// Maps render sets to its ending system (added during schedule build)
    pub(crate) render_sets_to_ending_systems: BTreeMap<SystemSetKey, SystemKey>,
}

/// Maps systems contained in submission sets to their submission state id.
#[derive(Debug, Default)]
pub(super) struct SubmissionSetsPass;

impl ScheduleBuildPass for SubmissionSetsPass {
    type EdgeOptions = ();

    fn add_dependency(
        &mut self,
        _from: bevy_ecs::schedule::NodeId,
        _to: bevy_ecs::schedule::NodeId,
        _options: Option<&Self::EdgeOptions>,
    ) {
    }

    fn collapse_set(
        &mut self,
        _set: SystemSetKey,
        _systems: &IndexSet<SystemKey, FixedHasher>,
        _dependency_flattening: &DiGraph<bevy_ecs::schedule::NodeId>,
    ) -> impl Iterator<Item = (bevy_ecs::schedule::NodeId, bevy_ecs::schedule::NodeId)> {
        // The registry is a World resource, while Bevy 0.18 calls collapse_set
        // before build and without World access. Registry-driven edges are
        // added directly to the flattened DAG in build instead.
        core::iter::empty()
    }

    fn build(
        &mut self,
        world: &mut World,
        graph: &mut ScheduleGraph,
        dependency_flattened: &mut Dag<SystemKey>,
    ) -> Result<(), ScheduleBuildError> {
        let Some(registry) = world.get_resource::<SubmissionSetRegistry>() else {
            return Ok(());
        };
        let submission_sets_to_queue = registry.submission_sets_to_queue.clone();
        let render_sets_to_systems = registry.render_sets_to_systems.clone();
        let mut submission_sets_to_meta_systems = registry.submission_sets_to_meta_systems.clone();
        let mut render_sets_to_ending_systems = registry.render_sets_to_ending_systems.clone();

        for render_set in render_sets_to_systems.keys().copied() {
            let Some(render_set_key) = graph.system_sets.get_key(render_set) else {
                continue;
            };
            render_sets_to_ending_systems
                .entry(render_set_key)
                .or_insert_with(|| {
                    let ending = add_system(graph, world, crate::system::render_set_ending_system);
                    dependency_flattened.add_node(ending);
                    ending
                });
        }

        for (submission_set, (queue_component_id, config)) in &submission_sets_to_queue {
            let Some(submission_set_key) = graph.system_sets.get_key(*submission_set) else {
                continue;
            };
            submission_sets_to_meta_systems
                .entry(submission_set_key)
                .or_insert_with(|| {
                    let state_id = world.register_component_with_descriptor(
                        ComponentDescriptor::new_resource::<crate::system::RenderSetSharedState>(),
                    );
                    let name = graph
                        .system_sets
                        .get(submission_set_key)
                        .map(|set| format!("{set:?}"))
                        .unwrap_or_else(|| format!("{submission_set:?}"));
                    let debug_color = config.debug_color;
                    let prelude_queue_component_id = *queue_component_id;
                    let submission_queue_component_id = *queue_component_id;

                    let prelude = add_system(graph, world, move |world: &mut World| {
                        crate::system::initialize_submission_state(
                            world,
                            state_id,
                            prelude_queue_component_id,
                            &name,
                            debug_color,
                        );
                        crate::system::prelude_system(world, state_id);
                    });
                    let submission = add_system(graph, world, move |world: &mut World| {
                        crate::system::submission_system(
                            world,
                            state_id,
                            submission_queue_component_id,
                        );
                    });

                    dependency_flattened.add_node(prelude);
                    dependency_flattened.add_node(submission);

                    RenderSetMetaSystems {
                        prelude,
                        submission,
                        state_id,
                    }
                });
        }

        {
            let mut registry = world.resource_mut::<SubmissionSetRegistry>();
            registry.submission_sets_to_meta_systems = submission_sets_to_meta_systems.clone();
            registry.render_sets_to_ending_systems = render_sets_to_ending_systems.clone();
        }

        world
            .resource_mut::<crate::system::SubmissionStates>()
            .clear_system_states();

        let hierarchy = graph.hierarchy().graph().clone();

        for (submission_set, _) in &submission_sets_to_queue {
            let Some(submission_set_key) = graph.system_sets.get_key(*submission_set) else {
                continue;
            };
            let Some(meta_systems) = submission_sets_to_meta_systems.get(&submission_set_key)
            else {
                continue;
            };

            let meta_set = HashSet::from([meta_systems.prelude, meta_systems.submission]);
            let mut render_systems = Vec::new();
            let mut nonrender_systems = Vec::new();
            collect_descendant_systems(
                &hierarchy,
                NodeId::Set(submission_set_key),
                &render_sets_to_ending_systems,
                false,
                &meta_set,
                &mut render_systems,
                &mut nonrender_systems,
            );

            let user_systems = render_systems
                .iter()
                .chain(&nonrender_systems)
                .copied()
                .collect::<Vec<_>>();

            for system_key in &user_systems {
                register_system_state_names(world, graph, *system_key, meta_systems.state_id);
            }

            for system in &user_systems {
                dependency_flattened.add_edge(meta_systems.prelude, *system);
                dependency_flattened.add_edge(*system, meta_systems.submission);
            }
            dependency_flattened.add_edge(meta_systems.prelude, meta_systems.submission);

            add_forwarded_set_dependencies(
                graph,
                dependency_flattened,
                submission_set_key,
                meta_systems.prelude,
                meta_systems.submission,
                &submission_sets_to_meta_systems,
                &render_sets_to_ending_systems,
            );
            add_forwarded_ancestor_set_dependencies(
                graph,
                dependency_flattened,
                &hierarchy,
                submission_set_key,
                meta_systems.prelude,
                meta_systems.submission,
                &submission_sets_to_meta_systems,
                &render_sets_to_ending_systems,
            );
        }

        for (render_set, config_system) in &render_sets_to_systems {
            let Some(render_set_key) = graph.system_sets.get_key(*render_set) else {
                continue;
            };
            let Some(ending_system) = render_sets_to_ending_systems.get(&render_set_key).copied()
            else {
                continue;
            };

            validate_render_set_parent(
                graph,
                render_set_key,
                *render_set,
                &submission_sets_to_queue,
            );

            let render_systems = collect_descendant_systems_flat(
                &hierarchy,
                NodeId::Set(render_set_key),
                &render_sets_to_ending_systems,
                true,
                &HashSet::new(),
            );

            for system in render_systems {
                if system != *config_system && system != ending_system {
                    dependency_flattened.add_edge(*config_system, system);
                    dependency_flattened.add_edge(system, ending_system);
                }
            }
            dependency_flattened.add_edge(*config_system, ending_system);

            add_forwarded_set_dependencies(
                graph,
                dependency_flattened,
                render_set_key,
                *config_system,
                ending_system,
                &submission_sets_to_meta_systems,
                &render_sets_to_ending_systems,
            );
        }

        for (submission_set, _) in &submission_sets_to_queue {
            let Some(submission_set_key) = graph.system_sets.get_key(*submission_set) else {
                continue;
            };
            let Some(meta_systems) = submission_sets_to_meta_systems.get(&submission_set_key)
            else {
                continue;
            };

            let meta_set = HashSet::from([meta_systems.prelude, meta_systems.submission]);
            let mut render_systems = Vec::new();
            let mut nonrender_systems = Vec::new();
            collect_descendant_systems(
                &hierarchy,
                NodeId::Set(submission_set_key),
                &render_sets_to_ending_systems,
                false,
                &meta_set,
                &mut render_systems,
                &mut nonrender_systems,
            );
            add_render_grouping_edges(dependency_flattened, &render_systems, &nonrender_systems);
        }

        Ok(())
    }
}

fn add_forwarded_set_dependencies(
    graph: &ScheduleGraph,
    dependency_flattened: &mut Dag<SystemKey>,
    set_key: SystemSetKey,
    entry_system: SystemKey,
    exit_system: SystemKey,
    submission_sets_to_meta_systems: &HashMap<SystemSetKey, RenderSetMetaSystems>,
    render_sets_to_ending_systems: &BTreeMap<SystemSetKey, SystemKey>,
) {
    let dependency = graph.dependency().graph();
    let hierarchy = graph.hierarchy().graph();
    let set_node = NodeId::Set(set_key);

    for parent in
        dependency.neighbors_directed(set_node, bevy_ecs::schedule::graph::Direction::Incoming)
    {
        for parent_system in dependency_source_systems(
            hierarchy,
            parent,
            submission_sets_to_meta_systems,
            render_sets_to_ending_systems,
        ) {
            dependency_flattened.add_edge(parent_system, entry_system);
        }
    }

    for child in
        dependency.neighbors_directed(set_node, bevy_ecs::schedule::graph::Direction::Outgoing)
    {
        for child_system in dependency_target_systems(
            hierarchy,
            child,
            submission_sets_to_meta_systems,
            render_sets_to_ending_systems,
        ) {
            dependency_flattened.add_edge(exit_system, child_system);
        }
    }
}

fn add_forwarded_ancestor_set_dependencies(
    graph: &ScheduleGraph,
    dependency_flattened: &mut Dag<SystemKey>,
    hierarchy: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    set_key: SystemSetKey,
    entry_system: SystemKey,
    exit_system: SystemKey,
    submission_sets_to_meta_systems: &HashMap<SystemSetKey, RenderSetMetaSystems>,
    render_sets_to_ending_systems: &BTreeMap<SystemSetKey, SystemKey>,
) {
    let mut to_visit = hierarchy
        .neighbors_directed(
            NodeId::Set(set_key),
            bevy_ecs::schedule::graph::Direction::Incoming,
        )
        .collect::<Vec<_>>();
    let mut visited = HashSet::new();

    while let Some(ancestor) = to_visit.pop() {
        if !visited.insert(ancestor) {
            continue;
        }

        let NodeId::Set(ancestor_set) = ancestor else {
            continue;
        };

        add_forwarded_set_dependencies(
            graph,
            dependency_flattened,
            ancestor_set,
            entry_system,
            exit_system,
            submission_sets_to_meta_systems,
            render_sets_to_ending_systems,
        );

        to_visit.extend(
            hierarchy.neighbors_directed(ancestor, bevy_ecs::schedule::graph::Direction::Incoming),
        );
    }
}

fn register_system_state_names(
    world: &mut World,
    graph: &mut ScheduleGraph,
    system_key: SystemKey,
    state_id: ComponentId,
) {
    if let Some(system) = graph.systems.get_mut(system_key) {
        system.access.add_unfiltered_resource_write(state_id);
    }

    if let Some(system) = graph.systems.get(system_key) {
        let name = system.name().to_string();
        let mut states = world.resource_mut::<crate::system::SubmissionStates>();
        states.set_system_state(name.clone(), state_id);

        if let Some((first, second)) = split_pipe_system_name(&name) {
            states.set_system_state(first.to_string(), state_id);
            states.set_system_state(second.to_string(), state_id);
        }
    }
}

fn split_pipe_system_name(name: &str) -> Option<(&str, &str)> {
    let inner = name.strip_prefix("Pipe(")?.strip_suffix(')')?;
    let mut angle_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;

    for (index, char) in inner.char_indices() {
        match char {
            '<' => angle_depth += 1,
            '>' => angle_depth = angle_depth.saturating_sub(1),
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            ',' if angle_depth == 0 && paren_depth == 0 && bracket_depth == 0 => {
                let first = inner[..index].trim();
                let second = inner[index + 1..].trim();
                return Some((first, second));
            }
            _ => {}
        }
    }

    None
}

fn dependency_source_systems(
    hierarchy: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    node: NodeId,
    submission_sets_to_meta_systems: &HashMap<SystemSetKey, RenderSetMetaSystems>,
    render_sets_to_ending_systems: &BTreeMap<SystemSetKey, SystemKey>,
) -> Vec<SystemKey> {
    match node {
        NodeId::System(system) => vec![system],
        NodeId::Set(set) => submission_sets_to_meta_systems
            .get(&set)
            .map(|meta| vec![meta.submission])
            .unwrap_or_else(|| {
                collect_descendant_systems_flat(
                    hierarchy,
                    node,
                    render_sets_to_ending_systems,
                    false,
                    &HashSet::new(),
                )
            }),
    }
}

fn dependency_target_systems(
    hierarchy: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    node: NodeId,
    submission_sets_to_meta_systems: &HashMap<SystemSetKey, RenderSetMetaSystems>,
    render_sets_to_ending_systems: &BTreeMap<SystemSetKey, SystemKey>,
) -> Vec<SystemKey> {
    match node {
        NodeId::System(system) => vec![system],
        NodeId::Set(set) => submission_sets_to_meta_systems
            .get(&set)
            .map(|meta| vec![meta.prelude])
            .unwrap_or_else(|| {
                collect_descendant_systems_flat(
                    hierarchy,
                    node,
                    render_sets_to_ending_systems,
                    false,
                    &HashSet::new(),
                )
            }),
    }
}

fn validate_render_set_parent(
    graph: &ScheduleGraph,
    render_set_key: SystemSetKey,
    render_set: InternedSystemSet,
    submission_sets_to_queue: &HashMap<InternedSystemSet, (ComponentId, SubmissionSetConfig)>,
) {
    let hierarchy = graph.hierarchy().graph();
    let mut count = 0usize;
    let mut to_visit = hierarchy
        .neighbors_directed(
            NodeId::Set(render_set_key),
            bevy_ecs::schedule::graph::Direction::Incoming,
        )
        .collect::<Vec<_>>();
    let mut visited = HashSet::new();

    while let Some(ancestor) = to_visit.pop() {
        if !visited.insert(ancestor) {
            continue;
        }
        if let NodeId::Set(ancestor_key) = ancestor {
            if submission_sets_to_queue
                .keys()
                .any(|set| graph.system_sets.get_key(*set) == Some(ancestor_key))
            {
                count += 1;
            }
            to_visit.extend(
                hierarchy
                    .neighbors_directed(ancestor, bevy_ecs::schedule::graph::Direction::Incoming),
            );
        }
    }

    if count == 0 {
        panic!(
            "Render set {render_set:?} is not inside any submission set. \
             Place it inside a submission set using `.in_set(your_submission_set)`."
        );
    }
    if count > 1 {
        panic!(
            "Render set {render_set:?} is inside {count} submission sets. \
             A render set must belong to exactly one submission set."
        );
    }
}

fn add_render_grouping_edges(
    dependency_flattened: &mut Dag<SystemKey>,
    render_systems: &[SystemKey],
    nonrender_systems: &[SystemKey],
) {
    if render_systems.is_empty() || nonrender_systems.is_empty() {
        return;
    }

    let render_set = render_systems.iter().copied().collect::<HashSet<_>>();
    let all_systems = render_systems
        .iter()
        .chain(nonrender_systems)
        .copied()
        .collect::<HashSet<_>>();

    let mut in_degree = all_systems
        .iter()
        .map(|&system| (system, 0usize))
        .collect::<HashMap<_, _>>();
    let mut successors = all_systems
        .iter()
        .map(|&system| (system, Vec::new()))
        .collect::<HashMap<_, _>>();

    for &system in &all_systems {
        for neighbor in dependency_flattened
            .neighbors_directed(system, bevy_ecs::schedule::graph::Direction::Outgoing)
        {
            if all_systems.contains(&neighbor) {
                successors.get_mut(&system).unwrap().push(neighbor);
                *in_degree.get_mut(&neighbor).unwrap() += 1;
            }
        }
    }

    let mut ready_render = Vec::new();
    let mut ready_nonrender = Vec::new();
    for (&system, &degree) in &in_degree {
        if degree == 0 {
            if render_set.contains(&system) {
                ready_render.push(system);
            } else {
                ready_nonrender.push(system);
            }
        }
    }

    let mut current_is_render = !ready_render.is_empty();
    let mut stages = Vec::<Vec<SystemKey>>::new();

    while !ready_render.is_empty() || !ready_nonrender.is_empty() {
        let queue = if current_is_render && !ready_render.is_empty() {
            &mut ready_render
        } else if !current_is_render && !ready_nonrender.is_empty() {
            &mut ready_nonrender
        } else {
            current_is_render = !current_is_render;
            continue;
        };

        if stages.last().is_none_or(|stage| {
            stage
                .first()
                .is_some_and(|system| render_set.contains(system))
                != current_is_render
        }) {
            stages.push(Vec::new());
        }

        let system = queue.pop().unwrap();
        stages.last_mut().unwrap().push(system);

        for &successor in &successors[&system] {
            let degree = in_degree.get_mut(&successor).unwrap();
            *degree -= 1;
            if *degree == 0 {
                if render_set.contains(&successor) {
                    ready_render.push(successor);
                } else {
                    ready_nonrender.push(successor);
                }
            }
        }
    }

    for pair in stages.windows(2) {
        for &from in &pair[0] {
            for &to in &pair[1] {
                add_edge_if_acyclic(dependency_flattened, from, to);
            }
        }
    }
}

fn add_edge_if_acyclic(graph: &mut Dag<SystemKey>, from: SystemKey, to: SystemKey) {
    if from == to || has_path(graph, to, from) {
        return;
    }
    graph.add_edge(from, to);
}

fn has_path(graph: &Dag<SystemKey>, from: SystemKey, to: SystemKey) -> bool {
    let mut stack = vec![from];
    let mut visited = HashSet::new();

    while let Some(system) = stack.pop() {
        if system == to {
            return true;
        }
        if !visited.insert(system) {
            continue;
        }
        stack.extend(
            graph.neighbors_directed(system, bevy_ecs::schedule::graph::Direction::Outgoing),
        );
    }

    false
}

fn collect_descendant_systems_flat(
    hierarchy: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    node: NodeId,
    render_sets_to_ending_systems: &BTreeMap<SystemSetKey, SystemKey>,
    in_render_set: bool,
    meta_systems: &HashSet<SystemKey>,
) -> Vec<SystemKey> {
    let mut render_systems = Vec::new();
    let mut nonrender_systems = Vec::new();
    collect_descendant_systems(
        hierarchy,
        node,
        render_sets_to_ending_systems,
        in_render_set,
        meta_systems,
        &mut render_systems,
        &mut nonrender_systems,
    );
    render_systems.extend(nonrender_systems);
    render_systems
}

/// Recursively walks the hierarchy graph from `node`, collecting descendant systems
/// into either `render_systems` or `nonrender_systems` based on whether they are
/// inside a render set.
fn collect_descendant_systems(
    hierarchy: &bevy_ecs::schedule::graph::DiGraph<NodeId>,
    node: NodeId,
    render_sets_to_ending_systems: &BTreeMap<SystemSetKey, SystemKey>,
    in_render_set: bool,
    meta_systems: &HashSet<SystemKey>,
    render_systems: &mut Vec<SystemKey>,
    nonrender_systems: &mut Vec<SystemKey>,
) {
    for child in hierarchy.neighbors_directed(node, bevy_ecs::schedule::graph::Direction::Outgoing)
    {
        match child {
            NodeId::System(sys) => {
                if meta_systems.contains(&sys) {
                    continue;
                }
                if in_render_set {
                    render_systems.push(sys);
                } else {
                    nonrender_systems.push(sys);
                }
            }
            NodeId::Set(set_key) => {
                let mut child_in_render = in_render_set;
                if let Some(ending_system) = render_sets_to_ending_systems.get(&set_key) {
                    render_systems.push(*ending_system);
                    child_in_render = true;
                }
                collect_descendant_systems(
                    hierarchy,
                    child,
                    render_sets_to_ending_systems,
                    child_in_render,
                    meta_systems,
                    render_systems,
                    nonrender_systems,
                );
            }
        }
    }
}

fn add_system<Marker, T: IntoSystem<(), (), Marker>>(
    graph: &mut ScheduleGraph,
    world: &mut World,
    system: T,
) -> SystemKey {
    let mut system: T::System = IntoSystem::into_system(system);
    let access = system.initialize(world);

    let id = graph.systems.insert(Box::new(system), Vec::new());

    // ignore ambiguities with auto sync points
    // They aren't under user control, so no one should know or care.
    graph.ambiguous_with_all.insert(id.into());
    graph.systems.get_mut(id).unwrap().access.extend(access);

    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::PostUpdate;
    use bevy_ecs::prelude::*;

    #[derive(Resource, Default)]
    struct DummyQueue;

    #[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct SubmissionSetA;

    #[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct SubmissionSetB;

    #[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct RenderSetA;

    #[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct SwapchainLikeSet;

    fn setup_system() {}
    fn config_system() {}
    fn render_system() {}
    fn present_system() {}
    fn submission_state_system_a(_state: crate::system::SubmissionState) {}
    fn submission_state_system_b(_state: crate::system::SubmissionState) {}

    #[test]
    fn splits_pipe_system_names_with_generic_commas() {
        let name = "Pipe(<bevy_app::app::App as bevy_pumicite::plugin::PumiciteApp>::add_render_set<(bevy_ecs::system::function_system::IsFunctionSystem, fn(bevy_pumicite::system::SubmissionState<'_>, bevy_ecs::system::query::Query<'_, '_, (&mut bevy_pumicite::swapchain::SwapchainImage, &mut gltf::GBuffer), bevy_ecs::query::filter::With<bevy_window::window::PrimaryWindow>>)), gltf::MainRenderPass, gltf::start_main_render_pass>::{{closure}}, gltf::start_main_render_pass)";
        let (first, second) = split_pipe_system_name(name).unwrap();

        assert_eq!(
            first,
            "<bevy_app::app::App as bevy_pumicite::plugin::PumiciteApp>::add_render_set<(bevy_ecs::system::function_system::IsFunctionSystem, fn(bevy_pumicite::system::SubmissionState<'_>, bevy_ecs::system::query::Query<'_, '_, (&mut bevy_pumicite::swapchain::SwapchainImage, &mut gltf::GBuffer), bevy_ecs::query::filter::With<bevy_window::window::PrimaryWindow>>)), gltf::MainRenderPass, gltf::start_main_render_pass>::{{closure}}"
        );
        assert_eq!(second, "gltf::start_main_render_pass");
    }

    #[test]
    fn builds_submission_and_render_set_graph() {
        let mut world = World::new();
        world.init_resource::<crate::system::SubmissionStates>();
        world.init_resource::<DummyQueue>();
        let queue_component_id = world
            .components()
            .resource_id::<DummyQueue>()
            .expect("DummyQueue should be registered");
        world.insert_resource(SubmissionSetRegistry::default());
        {
            let mut registry = world.resource_mut::<SubmissionSetRegistry>();
            registry.submission_sets_to_queue.insert(
                SubmissionSetA.intern(),
                (queue_component_id, SubmissionSetConfig::default()),
            );
            registry.submission_sets_to_queue.insert(
                SubmissionSetB.intern(),
                (queue_component_id, SubmissionSetConfig::default()),
            );
        }

        let mut schedule = Schedule::new(PostUpdate);
        schedule.add_build_pass(SubmissionSetsPass);
        schedule.configure_sets((
            SubmissionSetA
                .in_set(SwapchainLikeSet)
                .after(SubmissionSetB),
            SwapchainLikeSet.before(present_system),
            RenderSetA.in_set(SubmissionSetA),
        ));

        let config_key = {
            let existing_systems = schedule
                .graph()
                .systems
                .iter()
                .map(|(key, _, _)| key)
                .collect::<Vec<_>>();
            schedule.add_systems(config_system.in_set(RenderSetA));
            let added_systems = schedule
                .graph()
                .systems
                .iter()
                .map(|(key, _, _)| key)
                .filter(|key| !existing_systems.contains(key))
                .collect::<Vec<_>>();
            assert_eq!(added_systems.len(), 1);
            added_systems[0]
        };
        world
            .resource_mut::<SubmissionSetRegistry>()
            .render_sets_to_systems
            .insert(RenderSetA.intern(), config_key);

        schedule.add_systems((
            setup_system.in_set(SubmissionSetA).before(RenderSetA),
            render_system.in_set(RenderSetA).after(config_system),
            present_system,
        ));

        schedule.initialize(&mut world).unwrap();
    }

    #[test]
    fn submission_sets_use_distinct_state_resources() {
        let mut world = World::new();
        world.init_resource::<crate::system::SubmissionStates>();
        world.init_resource::<DummyQueue>();
        let queue_component_id = world
            .components()
            .resource_id::<DummyQueue>()
            .expect("DummyQueue should be registered");
        let registry_component_id = world
            .components()
            .resource_id::<crate::system::SubmissionStates>()
            .expect("SubmissionStates should be registered");
        world.insert_resource(SubmissionSetRegistry::default());
        {
            let mut registry = world.resource_mut::<SubmissionSetRegistry>();
            registry.submission_sets_to_queue.insert(
                SubmissionSetA.intern(),
                (queue_component_id, SubmissionSetConfig::default()),
            );
            registry.submission_sets_to_queue.insert(
                SubmissionSetB.intern(),
                (queue_component_id, SubmissionSetConfig::default()),
            );
        }

        let mut schedule = Schedule::new(PostUpdate);
        schedule.add_build_pass(SubmissionSetsPass);

        schedule.add_systems((
            submission_state_system_a.in_set(SubmissionSetA),
            submission_state_system_b.in_set(SubmissionSetB),
        ));

        schedule.initialize(&mut world).unwrap();

        let registry = world.resource::<SubmissionSetRegistry>();
        let submission_set_a = schedule
            .graph()
            .system_sets
            .get_key(SubmissionSetA.intern())
            .unwrap();
        let submission_set_b = schedule
            .graph()
            .system_sets
            .get_key(SubmissionSetB.intern())
            .unwrap();
        let state_a = registry
            .submission_sets_to_meta_systems
            .get(&submission_set_a)
            .unwrap()
            .state_id;
        let state_b = registry
            .submission_sets_to_meta_systems
            .get(&submission_set_b)
            .unwrap()
            .state_id;
        assert_ne!(state_a, state_b);
        assert_ne!(state_a, registry_component_id);
        assert_ne!(state_b, registry_component_id);
    }
}
