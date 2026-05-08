.PHONY: default
default: ray_tracing.spv

ray_tracing.spv: ray_tracing.slang
	slangc -g -O3 -target spirv -o ray_tracing.spv ray_tracing.slang
