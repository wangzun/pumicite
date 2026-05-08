use std::{
    alloc::Layout,
    collections::BTreeMap,
    ops::{Deref, RangeBounds},
    sync::Arc,
};

use ash::VkResult;
use bevy::{
    app::prelude::*,
    asset::prelude::*,
    ecs::prelude::*,
    transform::components::{GlobalTransform, Transform},
};
use bevy_pumicite::{
    loader::TextureAsset,
    prelude::*,
    rtx::{
        RayTracingPipeline, RtxPipelineManager, RtxPipelinePlugin,
        blas::{BLAS, BLASBuildGeometry, BLASBuilder, BLASBuilderPlugin},
        tlas::{TLAS, TLASBuilderPlugin, TLASBuilderSet, TLASInstance},
    },
};
use bytemuck::{AnyBitPattern, NoUninit, Pod, Zeroable};
use glam::{Mat4, UVec2, UVec3, Vec3, Vec3Swizzles};
use pumicite::{
    buffer::{Buffer, BufferLike, RingBufferSuballocation},
    image::{FullImageView, Image},
    utils::AsVkHandle,
};
use pumicite_scene::gltf::{self, GltfMaterialData};
use smallvec::SmallVec;

mod flycam;
use crate::flycam::{FlyCamera, FlyCameraPlugin};

fn main() {
    let mut app = App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins)
        .add_plugins(RtxPipelinePlugin)
        .add_plugins(BLASBuilderPlugin::<GltfModelBlasBuilder>::default())
        .add_plugins(TLASBuilderPlugin::<RayTracingInstanceData>::default())
        .add_plugins(pumicite_scene::gltf::GltfPlugin)
        .add_plugins(FlyCameraPlugin);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut().entity_mut(primary_window).insert((
        SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        RayTarget::default(),
    ));

    app.world_mut().spawn((
        GlobalTransform::default(),
        Transform::from_translation(Vec3::new(0.0, 0.5, 2.0)),
        FlyCamera {
            max_speed: 1.0,
            accel: 1.0,
            friction: 1.0,
            ..Default::default()
        },
    ));

    app.enable_feature::<vk::PhysicalDeviceFeatures>(|x| &mut x.shader_int64)
        .unwrap();
    app.add_device_extension::<ash::khr::push_descriptor::Meta>()
        .unwrap();
    app.enable_bindless().unwrap();

    app.add_systems(Startup, setup.after(bevy_pumicite::CreateDevice));
    app.add_systems(
        PostUpdate,
        (
            ray_target_resize.in_set(DefaultRenderSet),
            prepare_ray_scene
                .in_set(DefaultTransferSet)
                .before(TLASBuilderSet::<RayTracingInstanceData>::default()),
            sync_ray_instances.in_set(TLASBuilderSet::<RayTracingInstanceData>::default()),
            trace_gltf_scene.in_set(DefaultRenderSet),
        ),
    );
    app.run();
}

#[derive(Resource)]
struct RayTracingExample {
    pipeline: Handle<RayTracingPipeline>,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut rtx_pipelines: ResMut<RtxPipelineManager>,
) {
    let scene: Handle<bevy::scene::Scene> = asset_server.load("gltf/FlightHelmet.gltf");
    commands.spawn(bevy::scene::SceneRoot(scene));

    let base_library = asset_server.load("ray_tracing/ray_tracing.rtx.pipeline.ron");
    let pipeline = rtx_pipelines.add_pipeline(base_library);
    commands.insert_resource(RayTracingExample { pipeline });
    commands.insert_resource(PreparedRayScene::default());
}

#[derive(Component, Default)]
struct RayTarget {
    texture: Option<GPUMutex<RayTargetTexture>>,
    state: ResourceState,
    extent: UVec2,
}

struct RayTargetTexture {
    color: FullImageView<Image>,
}

impl RayTargetTexture {
    fn new(allocator: Allocator, extent: UVec2) -> VkResult<Self> {
        let color = Image::new_private(
            allocator,
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: extent.x,
                    height: extent.y,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
        )?;
        Ok(Self {
            color: color.create_full_view()?,
        })
    }
}

fn ray_target_resize(
    mut targets: Query<(&SwapchainImage, &mut RayTarget), With<bevy::window::PrimaryWindow>>,
    allocator: Res<Allocator>,
) {
    let Ok((swapchain_image, mut target)) = targets.single_mut() else {
        return;
    };
    let Some(current_swapchain_image) = swapchain_image.current_image() else {
        return;
    };
    let extent = current_swapchain_image.extent().xy();
    let old_extent = target
        .texture
        .as_ref()
        .map(|texture| texture.color.image().extent().xy());
    if old_extent != Some(extent) {
        target.texture = Some(GPUMutex::new(
            RayTargetTexture::new(allocator.clone(), extent).unwrap(),
        ));
        target.state = ResourceState::default();
        target.extent = extent;
    }
}

#[derive(Default, Resource)]
struct GltfModelBlasBuilder;

impl BLASBuilder for GltfModelBlasBuilder {
    type QueryData = &'static pumicite_scene::Model;
    type QueryFilter = Without<BLAS>;
    type Params = ();
    type BufferType = BufferView;

    fn geometries<'w, 's, 't, 't2, 'b, 'bb>(
        &mut self,
        _params: &mut bevy::ecs::system::SystemParamItem<'w, 's, Self::Params>,
        model: bevy::ecs::query::QueryItem<'t, 't2, Self::QueryData>,
        recorder: &'bb mut pumicite::command::CommandEncoder<'b>,
    ) -> impl Future<Output = SmallVec<[BLASBuildGeometry<'b, Self::BufferType>; 1]>>
    + use<'w, 's, 't, 't2, 'b, 'bb> {
        async move {
            let mut geometries = SmallVec::new();
            for primitive in model.iter() {
                if primitive.topology != vk::PrimitiveTopology::TRIANGLE_LIST {
                    continue;
                }
                let Some(position) = primitive.attribute(gltf::Semantic::Positions) else {
                    continue;
                };
                let Some(index_buffer) = primitive.index_buffer.as_ref() else {
                    continue;
                };
                let vertex_data = recorder.retain(BufferView {
                    buffer: position.buffer.clone(),
                    offset: position.offset as u64,
                });
                let index_data = recorder.retain(BufferView {
                    buffer: index_buffer.clone(),
                    offset: primitive.index_buffer_offset,
                });
                geometries.push(BLASBuildGeometry::Triangles {
                    vertex_format: vk::Format::R32G32B32_SFLOAT,
                    vertex_data,
                    vertex_stride: position.stride as u64,
                    max_vertex: position.count.saturating_sub(1) as u32,
                    index_type: acceleration_index_type(primitive.index_type),
                    index_data,
                    transform_data: None,
                    flags: vk::GeometryFlagsKHR::OPAQUE,
                    primitive_count: primitive.index_count / 3,
                });
            }
            geometries
        }
    }
}

#[derive(Clone)]
struct BufferView {
    buffer: Arc<Buffer>,
    offset: u64,
}

impl AsVkHandle for BufferView {
    type Handle = vk::Buffer;

    fn vk_handle(&self) -> Self::Handle {
        self.buffer.vk_handle()
    }
}

impl BufferLike for BufferView {
    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    fn device_address(&self) -> vk::DeviceAddress {
        self.buffer.device_address() + self.offset
    }

    fn size(&self) -> vk::DeviceSize {
        self.buffer.size() - self.offset
    }

    fn as_slice(&self) -> Option<&[u8]> {
        None
    }

    fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        None
    }

    fn flush(&mut self, _range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        Ok(())
    }

    fn invalidate(&mut self, _range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        Ok(())
    }
}

#[derive(NoUninit, Clone, Copy, AnyBitPattern)]
#[repr(C)]
struct PrimitiveData {
    p_positions: u64,
    p_normals: u64,
    p_texcoords: u64,
    p_indices: u64,
    material_index: u32,
    index_type: u32,
    _padding: [u32; 2],
}

#[derive(NoUninit, Clone, Copy, AnyBitPattern)]
#[repr(C)]
struct ModelInfo {
    primitive_offset: u32,
    primitive_count: u32,
    _padding: [u32; 2],
}

#[derive(NoUninit, Clone, Copy, AnyBitPattern)]
#[repr(C)]
struct CameraUniforms {
    view: Mat4,
    projection: Mat4,
    inv_view_projection: Mat4,
    position: [f32; 4],
}

#[derive(Zeroable, Pod, Clone, Copy, bevy_reflect::TypePath)]
#[repr(C)]
struct RayTracingInstanceData {
    transform: [[f32; 4]; 4],
    model_index: u32,
    _padding: [u32; 3],
}

#[derive(Default, Resource)]
struct PreparedRayScene {
    camera_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    material_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    primitive_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    model_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    sbt_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    sbt: Option<pumicite::rtx::ShaderBindingTable>,
    model_mapping: BTreeMap<Entity, u32>,
}

fn prepare_ray_scene(
    cameras: Query<&GlobalTransform, With<FlyCamera>>,
    models: Query<(
        Entity,
        &pumicite_scene::Model,
        &pumicite_scene::ModelInstances,
    )>,
    materials: Query<(Entity, &pumicite_scene::gltf::GltfMaterial)>,
    textures: Res<Assets<TextureAsset>>,
    mut ring_buffer: BufferInitializer,
    mut ctx: SubmissionState,
    mut prepared_scene: ResMut<PreparedRayScene>,
    targets: Query<&RayTarget, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(camera_transform) = cameras.single() else {
        return;
    };
    let Ok(ray_target) = targets.single() else {
        return;
    };
    ctx.record(|encoder| {
        let aspect =
            ray_target.extent.x as f32 / ray_target.extent.y as f32;
        let view = camera_transform.to_matrix().inverse();
        let projection =
            Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_3, aspect, 0.1);
        prepared_scene.camera_buffer = Some(ring_buffer.create_preinitialized_buffer(
            encoder,
            Layout::new::<CameraUniforms>(),
            |dst| {
                let camera: &mut CameraUniforms = bytemuck::from_bytes_mut(dst);
                camera.view = view;
                camera.projection = projection;
                camera.inv_view_projection = (projection * view).inverse();
                camera.position = camera_transform.translation().extend(1.0).to_array();
            },
        ));

        let num_materials = materials.iter().len();
        let mut material_mapping = BTreeMap::new();
        prepared_scene.material_buffer = Some(
            ring_buffer.create_preinitialized_buffer(
                encoder,
                Layout::new::<GltfMaterialData>()
                    .repeat(num_materials)
                    .unwrap()
                    .0,
                |dst| {
                    let dst: &mut [GltfMaterialData] = bytemuck::cast_slice_mut(dst);
                    for (index, ((entity, material), dst)) in materials.iter().zip(dst).enumerate()
                    {
                        material_mapping.insert(entity, index as u32);
                        *dst = GltfMaterialData {
                            base_color_factor: material.base_color_factor,
                            base_color: material
                                .base_color
                                .as_ref()
                                .and_then(|x| textures.get(&x.image))
                                .map(|x| x.handle())
                                .unwrap_or(u32::MAX),
                            base_color_sampler: material
                                .base_color
                                .as_ref()
                                .map(|x| x.sampler.id())
                                .unwrap_or(u32::MAX),
                            _padding: [0; 2],
                        };
                    }
                },
            ),
        );

        let num_models = models.iter().len();
        let num_primitives: usize = models.iter().map(|(_, model, _)| model.len()).sum();
        let mut model_mapping = BTreeMap::new();
        let mut primitive_offset = 0u32;
        prepared_scene.model_buffer = Some(ring_buffer.create_preinitialized_buffer(
            encoder,
            Layout::new::<ModelInfo>().repeat(num_models).unwrap().0,
            |dst| {
                let dst: &mut [ModelInfo] = bytemuck::cast_slice_mut(dst);
                for ((entity, model, _), dst) in models.iter().zip(dst) {
                    model_mapping.insert(entity, primitive_offset);
                    *dst = ModelInfo {
                        primitive_offset,
                        primitive_count: model.len() as u32,
                        _padding: [0; 2],
                    };
                    primitive_offset += model.len() as u32;
                }
            },
        ));
        prepared_scene.model_mapping = model_mapping;

        prepared_scene.primitive_buffer = Some(
            ring_buffer.create_preinitialized_buffer(
                encoder,
                Layout::new::<PrimitiveData>()
                    .repeat(num_primitives)
                    .unwrap()
                    .0,
                |dst| {
                    let dst: &mut [PrimitiveData] = bytemuck::cast_slice_mut(dst);
                    for (primitive, dst) in models
                        .iter()
                        .flat_map(|(_, model, _)| model.iter())
                        .zip(dst)
                    {
                        *dst = PrimitiveData {
                            p_positions: primitive.attribute_gpuva(gltf::Semantic::Positions),
                            p_normals: primitive.attribute_gpuva(gltf::Semantic::Normals),
                            p_texcoords: primitive.attribute_gpuva(gltf::Semantic::TexCoords(0)),
                            p_indices: primitive
                                .index_buffer
                                .as_ref()
                                .map(|buffer| {
                                    buffer.device_address() + primitive.index_buffer_offset
                                })
                                .unwrap_or_default(),
                            material_index: material_mapping
                                .get(&primitive.material)
                                .copied()
                                .unwrap_or(u32::MAX),
                            index_type: shader_index_type(primitive.index_type),
                            _padding: [0; 2],
                        };
                    }
                },
            ),
        );
    });
}

fn sync_ray_instances(
    mut commands: Commands,
    blas: Query<&BLAS>,
    mut instances: Query<(
        Entity,
        &pumicite_scene::InstanceOf,
        &GlobalTransform,
        Option<&mut TLASInstance<RayTracingInstanceData>>,
    )>,
    prepared_scene: Res<PreparedRayScene>,
) {
    for (entity, instance_of, transform, instance) in instances.iter_mut() {
        let Some(&model_index) = prepared_scene.model_mapping.get(&instance_of.model) else {
            continue;
        };
        let data = RayTracingInstanceData {
            transform: transform.to_matrix().to_cols_array_2d(),
            model_index,
            _padding: [0; 3],
        };
        let disabled = !blas.contains(instance_of.model);
        if let Some(mut instance) = instance {
            instance.blas = instance_of.model;
            instance.disabled = disabled;
            instance.data = data;
        } else {
            let mut instance = TLASInstance::new(instance_of.model);
            instance.disabled = disabled;
            instance.data = data;
            commands.entity(entity).insert(instance);
        }
    }
}

fn trace_gltf_scene(
    mut swapchain: Query<(&mut SwapchainImage, &mut RayTarget), With<bevy::window::PrimaryWindow>>,
    mut ctx: SubmissionState,
    mut ring_buffer: BufferInitializer,
    mut prepared_scene: ResMut<PreparedRayScene>,
    ray_tracing_example: Res<RayTracingExample>,
    pipelines: Res<Assets<RayTracingPipeline>>,
    tlas: Res<TLAS<RayTracingInstanceData>>,
    heap: Res<DescriptorHeap>,
) {
    let Ok((mut swapchain_image, mut target)) = swapchain.single_mut() else {
        return;
    };
    let Some(pipeline) = pipelines.get(&ray_tracing_example.pipeline) else {
        return;
    };
    let Some(tlas_inner) = tlas.get() else {
        return;
    };
    ctx.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };

        let Some(target_texture) = target.texture.as_ref() else {
            return;
        };
        let mut sbt = pipeline.create_sbt(prepared_scene.sbt.take());
        sbt.push_raygen(0, |_| {});
        sbt.push_miss(0, |_| {});
        sbt.push_hitgroup(0, |_| {});
        prepared_scene.sbt_buffer = Some(ring_buffer.create_preinitialized_buffer(
            encoder,
            sbt.layout(),
            |dst| dst.copy_from_slice(sbt.buffer()),
        ));
        prepared_scene.sbt = Some(sbt);

        let (Some(camera), Some(materials), Some(primitives), Some(models), Some(sbt_buffer)) = (
            prepared_scene.camera_buffer.as_ref(),
            prepared_scene.material_buffer.as_ref(),
            prepared_scene.primitive_buffer.as_ref(),
            prepared_scene.model_buffer.as_ref(),
            prepared_scene.sbt_buffer.as_ref(),
        ) else {
            return;
        };

        let tlas_inner = encoder.lock(tlas_inner, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let target_texture = encoder.lock(
            target_texture,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::BLIT,
        );
        let current_swapchain_image =
            encoder.lock(current_swapchain_image, vk::PipelineStageFlags2::BLIT);

        let camera = encoder.lock(camera, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let materials = encoder.lock(materials, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let primitives = encoder.lock(primitives, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let models = encoder.lock(models, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let sbt_buffer = encoder.lock(sbt_buffer, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);

        encoder.use_image_resource(
            target_texture.color.image(),
            &mut target.state,
            Access::RTX_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let acceleration_structures = [tlas_inner.vk_handle()];
        let mut acceleration_structure_info = vk::WriteDescriptorSetAccelerationStructureKHR {
            acceleration_structure_count: 1,
            p_acceleration_structures: acceleration_structures.as_ptr(),
            ..Default::default()
        };
        let output_image_info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: target_texture.color.vk_handle(),
            image_layout: vk::ImageLayout::GENERAL,
        };
        let writes = [
            vk::WriteDescriptorSet {
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &output_image_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                p_next: (&mut acceleration_structure_info
                    as *mut vk::WriteDescriptorSetAccelerationStructureKHR)
                    .cast(),
                ..Default::default()
            },
        ];

        let pipeline = encoder.retain(pipeline.deref().clone());
        let pipeline_ref = pipeline.as_ref();
        encoder.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline_ref);
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline_ref.layout(),
            0,
            &writes,
        );
        encoder.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline_ref.layout(),
            1,
            &[
                heap.resource_heap().descriptor_set(),
                heap.sampler_heap().descriptor_set(),
            ],
            &[],
        );
        let Some(instance_data) = tlas.tlas_per_instance_data.as_ref() else {
            return;
        };

        let instance_data = encoder.lock(
            instance_data,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        encoder.push_constants(
            pipeline_ref.layout(),
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&PushConstants {
                p_primitives: primitives.device_address(),
                p_models: models.device_address(),
                p_instances: instance_data.device_address(),
                p_materials: materials.device_address(),
                p_camera: camera.device_address(),
                sun_direction: Vec3::new(-0.4, -0.8, -0.3).normalize().to_array(),
                _padding: 0,
            }),
        );

        let dispatch_extent = target_texture.color.image().extent();
        encoder.trace_rays(
            prepared_scene.sbt.as_ref().unwrap(),
            0,
            sbt_buffer,
            UVec3::new(dispatch_extent.x, dispatch_extent.y, 1),
        );

        encoder.use_image_resource(
            target_texture.color.image(),
            &mut target.state,
            Access::BLIT_SRC,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::BLIT_DST,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        encoder.blit_image_with_layout(
            target_texture.color.image(),
            vk::ImageLayout::GENERAL,
            current_swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                src_offsets: [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: dispatch_extent.x as i32,
                        y: dispatch_extent.y as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                dst_offsets: [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: current_swapchain_image.extent().x as i32,
                        y: current_swapchain_image.extent().y as i32,
                        z: 1,
                    },
                ],
            }],
            vk::Filter::NEAREST,
        );
    });
}

#[derive(NoUninit, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    p_primitives: u64,
    p_models: u64,
    p_instances: u64,
    p_materials: u64,
    p_camera: u64,
    sun_direction: [f32; 3],
    _padding: u32,
}

fn acceleration_index_type(index_type: vk::IndexType) -> vk::IndexType {
    match index_type {
        vk::IndexType::UINT8_EXT => vk::IndexType::UINT16,
        other => other,
    }
}

fn shader_index_type(index_type: vk::IndexType) -> u32 {
    match acceleration_index_type(index_type) {
        vk::IndexType::UINT16 => 0,
        vk::IndexType::UINT32 => 1,
        vk::IndexType::UINT8_EXT => 2,
        _ => 0,
    }
}
