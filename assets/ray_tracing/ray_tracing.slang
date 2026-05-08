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
    uint samplesPerPixel;
    uint maxBounces;
    uint frameIndex;
    uint2 _padding;
};

[vk::push_constant]
PushConstants pc;

[vk::binding(1, 0)]
RaytracingAccelerationStructure scene;

[format("rgba8")]
[vk::binding(0, 0)]
RWTexture2D<float4> outputImage;

struct RayPayload {
    uint hit;
    float3 color;
    float3 albedo;
    float3 normal;
    float3 position;
};

uint pcgHash(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand(inout uint state)
{
    return float(pcgHash(state)) * (1.0 / 4294967296.0);
}

float3 skyColor(float3 direction)
{
    float t = saturate(direction.y * 0.5 + 0.5);
    return lerp(float3(0.03, 0.04, 0.06), float3(0.55, 0.70, 0.95), t);
}

float3 cosineSampleHemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 6.28318530718 * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

float3 toWorldHemisphere(float3 localDirection, float3 normal)
{
    float3 helper = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(helper, normal));
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * localDirection.x + bitangent * localDirection.y + normal * localDirection.z);
}

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
    uint sampleCount = max(pc.samplesPerPixel, 1u);
    float3 color = float3(0.0);

    for (uint sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        uint rng = pixel.x * 1973u
            ^ pixel.y * 9277u
            ^ pc.frameIndex * 26699u
            ^ sampleIndex * 104729u;

        float2 jitter = float2(rand(rng), rand(rng));
        float2 uv = (float2(pixel) + jitter) / float2(extent);
        float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

        // The example camera uses a reversed-Z infinite projection. Clip-space z = 0
        // is the infinite far plane, so unproject the near plane instead.
        float4 world = mul(pc.camera.invViewProjection, float4(ndc, 1.0, 1.0));
        world.xyz /= world.w;

        RayDesc ray;
        ray.Origin = pc.camera.position.xyz;
        ray.Direction = normalize(world.xyz - ray.Origin);
        ray.TMin = 0.001;
        ray.TMax = 10000.0;

        float3 radiance = float3(0.0);
        float3 throughput = float3(1.0);

        for (uint bounce = 0; bounce <= pc.maxBounces; bounce++)
        {
            RayPayload payload;
            payload.hit = 0u;
            payload.color = float3(0.0);
            payload.albedo = float3(0.0);
            payload.normal = float3(0.0, 1.0, 0.0);
            payload.position = ray.Origin;

            TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);

            if (payload.hit == 0u)
            {
                radiance += throughput * payload.color;
                break;
            }

            if (bounce == pc.maxBounces)
            {
                break;
            }

            throughput *= payload.albedo;

            if (bounce >= 2u)
            {
                float p = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05, 0.95);
                if (rand(rng) > p)
                {
                    break;
                }
                throughput /= p;
            }

            float3 localDirection = cosineSampleHemisphere(float2(rand(rng), rand(rng)));
            ray.Direction = toWorldHemisphere(localDirection, payload.normal);
            ray.Origin = payload.position + payload.normal * 0.002;
            ray.TMin = 0.001;
            ray.TMax = 10000.0;
        }

        color += radiance;
    }

    color /= float(sampleCount);
    color = color / (color + float3(1.0));
    color = pow(max(color, float3(0.0)), float3(1.0 / 2.2));
    outputImage[pixel] = float4(color, 1.0);
}

[shader("miss")]
void missMain(inout RayPayload payload)
{
    payload.hit = 0u;
    payload.color = skyColor(WorldRayDirection());
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
    if (dot(worldNormal, WorldRayDirection()) > 0.0)
    {
        worldNormal = -worldNormal;
    }

    float3 p0 = primitive.positions[i0];
    float3 p1 = primitive.positions[i1];
    float3 p2 = primitive.positions[i2];
    float3 localPosition = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    float3 worldPosition = mul(instance.transform, float4(localPosition, 1.0)).xyz;

    float2 uv0 = primitive.texcoords != nullptr ? primitive.texcoords[i0] : float2(0.0, 0.0);
    float2 uv1 = primitive.texcoords != nullptr ? primitive.texcoords[i1] : uv0;
    float2 uv2 = primitive.texcoords != nullptr ? primitive.texcoords[i2] : uv0;
    float2 texcoord = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    GltfMaterialData material = pc.materials[primitive.materialIndex];
    float3 albedo = sampleBaseColor(material, texcoord);
    payload.hit = 1u;
    payload.color = float3(0.0);
    payload.albedo = clamp(albedo, float3(0.0), float3(0.98));
    payload.normal = worldNormal;
    payload.position = worldPosition;
}
