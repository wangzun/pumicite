//! Compatibility build pass for Bevy schedule construction.

use bevy_ecs::{
    resource::Resource,
    schedule::{
        InternedSystemSet, NodeId, ScheduleBuildError, ScheduleBuildPass, ScheduleGraph, SystemKey,
        SystemSetKey,
        graph::{Dag, DiGraph},
    },
    system::System,
    world::World,
};
use bevy_platform::hash::FixedHasher;
use indexmap::IndexSet;

#[derive(Resource, Default)]
pub(crate) struct SubmissionSetRegistry {
    sets: Vec<(InternedSystemSet, usize)>,
}

impl SubmissionSetRegistry {
    pub(crate) fn register(&mut self, set: InternedSystemSet, state_id: usize) {
        self.sets.push((set, state_id));
    }
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
        core::iter::empty()
    }

    fn build(
        &mut self,
        world: &mut World,
        graph: &mut ScheduleGraph,
        _dependency_flattened: &mut Dag<SystemKey>,
    ) -> Result<(), ScheduleBuildError> {
        let Some(registry) = world.get_resource::<SubmissionSetRegistry>() else {
            return Ok(());
        };
        let registered_sets = registry.sets.clone();
        world
            .resource_mut::<crate::system::SubmissionStates>()
            .clear_system_states();

        for (set, state_id) in registered_sets {
            let Some(set_key) = graph.system_sets.get_key(set) else {
                continue;
            };
            let mut stack = vec![NodeId::Set(set_key)];
            while let Some(node) = stack.pop() {
                for child in graph
                    .hierarchy()
                    .neighbors_directed(node, bevy_ecs::schedule::graph::Direction::Outgoing)
                {
                    match child {
                        NodeId::System(system_key) => {
                            if let Some(system) = graph.systems.get(system_key) {
                                world
                                    .resource_mut::<crate::system::SubmissionStates>()
                                    .set_system_state(system.name().to_string(), state_id);
                            }
                        }
                        NodeId::Set(_) => stack.push(child),
                    }
                }
            }
        }
        Ok(())
    }
}
