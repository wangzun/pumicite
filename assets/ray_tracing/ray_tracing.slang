[vk::binding(0, 1)]
__DynamicResource<__DynamicResourceKind.General> resourceHandles[];

[vk::binding(0, 2)]
__DynamicResource<__DynamicResourceKind.Sampler> samplerHandles[];

export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    __target_switch
    {
    case spirv:
        if (T.kind == DescriptorKind.Sampler)
            return samplerHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
        else
            return resourceHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
    default:
        return defaultGetDescriptorFromHandle(handle);
    }
}

struct GltfMaterialData {
    float4 baseColorFactor;
    uint32_t baseColorTextureIndex;
    uint32_t baseColorSamplerIndex;
    uint32_t _padding[2];
};

struct PrimitiveData {
    float3* positions;
    float3* normals;
    float2* texcoords;
    uint8_t* indices;
    uint materialIndex;
    uint indexType;
    uint2 _padding;
};

struct ModelInfo {
    uint primitiveOffset;
    uint primitiveCount;
    uint2 _padding;
};

struct InstanceData {
    float4x4 transform;
    uint modelIndex;
    uint3 _padding;
};

struct CameraUniforms {
    float4x4 view;
    float4x4 projection;
    float4x4 invViewProjection;
    float4 position;
};

struct PushConstants {
    PrimitiveData* primitives;
    ModelInfo* models;
    InstanceData* instances;
    GltfMaterialData* materials;
    CameraUniforms* camera;
    float3 sunDirection;
    uint _padding;
};

[vk::push_constant]
PushConstants pc;

[vk::binding(0, 0)]
RaytracingAccelerationStructure scene;

[format("rgba8")]
[vk::binding(1, 0)]
RWTexture2D<float4> outputImage;

struct RayPayload {
    float3 color;
};

uint loadIndex(PrimitiveData primitive, uint offset)
{
    if (primitive.indexType == 1)
    {
        uint32_t* indices = (uint32_t*)primitive.indices;
        return indices[offset];
    }
    if (primitive.indexType == 2)
    {
        return primitive.indices[offset];
    }
    uint16_t* indices = (uint16_t*)primitive.indices;
    return indices[offset];
}

float3 sampleBaseColor(GltfMaterialData material, float2 uv)
{
    float4 baseColor = material.baseColorFactor;
    if (material.baseColorTextureIndex != ~0u)
    {
        Texture2D.Handle baseColorTexture = (Texture2D.Handle)material.baseColorTextureIndex;
        SamplerState.Handle baseColorSampler = (SamplerState.Handle)material.baseColorSamplerIndex;
        baseColor *= baseColorTexture.SampleLevel(baseColorSampler, uv, 0.0);
    }
    return baseColor.xyz;
}

[shader("raygeneration")]
void rayGenMain()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 extent = DispatchRaysDimensions().xy;
    float2 uv = (float2(pixel) + 0.5) / float2(extent);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4 world = mul(pc.camera.invViewProjection, float4(ndc, 0.0, 1.0));
    world.xyz /= world.w;

    RayDesc ray;
    ray.Origin = pc.camera.position.xyz;
    ray.Direction = normalize(world.xyz - ray.Origin);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload;
    payload.color = float3(0.55, 0.70, 0.95);
    TraceRay(scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 0, 0, ray, payload);

    outputImage[pixel] = float4(payload.color, 1.0);
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    float t = saturate(WorldRayDirection().y * 0.5 + 0.5);
    payload.color = lerp(float3(0.03, 0.04, 0.06), float3(0.55, 0.70, 0.95), t);
}

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attr)
{
    InstanceData instance = pc.instances[InstanceIndex()];
    ModelInfo model = pc.models[instance.modelIndex];
    PrimitiveData primitive = pc.primitives[model.primitiveOffset + GeometryIndex()];

    uint indexOffset = PrimitiveIndex() * 3;
    uint i0 = loadIndex(primitive, indexOffset + 0);
    uint i1 = loadIndex(primitive, indexOffset + 1);
    uint i2 = loadIndex(primitive, indexOffset + 2);

    float3 bary = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    float3 n0 = primitive.normals != nullptr ? primitive.normals[i0] : float3(0.0, 1.0, 0.0);
    float3 n1 = primitive.normals != nullptr ? primitive.normals[i1] : n0;
    float3 n2 = primitive.normals != nullptr ? primitive.normals[i2] : n0;
    float3 localNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
    float3 worldNormal = normalize(mul((float3x3)instance.transform, localNormal));

    float2 uv0 = primitive.texcoords != nullptr ? primitive.texcoords[i0] : float2(0.0, 0.0);
    float2 uv1 = primitive.texcoords != nullptr ? primitive.texcoords[i1] : uv0;
    float2 uv2 = primitive.texcoords != nullptr ? primitive.texcoords[i2] : uv0;
    float2 texcoord = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    GltfMaterialData material = pc.materials[primitive.materialIndex];
    float3 albedo = sampleBaseColor(material, texcoord);
    float nDotL = saturate(dot(worldNormal, -pc.sunDirection));
    payload.color = albedo * (0.08 + 0.92 * nDotL);
}
