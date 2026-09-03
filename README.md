# SeedVR2 NCNN

This repository contains only the SeedVR2 image/video forward runtime. It
includes the DiT, VAE, pipeline, Vulkan custom layers, media I/O, and one
production command-line executable. It does not include model weights, export
scripts, or the other examples from `ncnn_llm`.

The runtime supports the exported SeedVR2 frontend, 32 DiT blocks, tail, VAE
encoding/decoding, and image/video input/output. The DiT and VAE model
directories are supplied at runtime, so weights remain external.

## Requirements

- Linux with a C++17 compiler
- CMake 3.18 or newer
- Vulkan headers, loader, and a working GPU driver for inference
- FFmpeg development packages (`libavformat`, `libavcodec`, `libavutil`, and
  `libswscale`)
- Git with submodule support

On Ubuntu/Debian, install the build dependencies with:

```bash
sudo apt install build-essential cmake pkg-config libvulkan-dev \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
```

## Build

Clone this repository with its ncnn submodule:

```bash
git clone --recurse-submodules <repository-url> seedvr2-ncnn
cd seedvr2-ncnn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The build accepts only ncnn commit
`946fe3fb14a8dff8c06df763f67be522167b2f00`. During CMake configuration it
applies the two patches in `patches/ncnn/` required by the SeedVR2 Vulkan
runtime. Re-running CMake is safe.

The executable is generated at:

```text
build/seedvr2_main
```

## Run

Load exported SeedVR2 DiT and VAE directories:

```bash
./build/seedvr2_main \
  --dit /path/to/seedvr2_ncnn_dit \
  --vae /path/to/seedvr2_ncnn_vae \
  --video-path input.png \
  --output-dir output \
  --text-pos positive.f32 \
  --gpu 0 --steps 1 --cfg-scale 1
```

For a video input, use the same command with a video path. The output format is
selected by the input type: image inputs are written as PNG and video inputs
are encoded by FFmpeg.

```bash
./build/seedvr2_main \
  --dit /path/to/seedvr2_ncnn_dit \
  --vae /path/to/seedvr2_ncnn_vae \
  --video-path input.mp4 \
  --output-dir output \
  --text-pos positive.f32 \
  --gpu 0 --steps 1 --cfg-scale 1
```

The expected DiT model directory contains `frontend/`, `block_00/` through
`block_31/`, and `tail/`, each with its `.ncnn.param` and `.ncnn.bin` files.
The VAE model directory contains `seedvr2_vae_encoder.ncnn.param/.bin` and
`seedvr2_vae_decoder.ncnn.param/.bin`.

## ncnn version

The ncnn submodule is pinned to commit
`946fe3fb14a8dff8c06df763f67be522167b2f00`. Do not update it independently:
the SeedVR2 patches and the exported runtime depend on this ABI and shader
implementation.
