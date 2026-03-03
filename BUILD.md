# Build Instructions

## Prerequisites

- **CMake** >= 3.10
- **MinGW** (GCC) with C++11 support
- **SFML** >= 2.5 (graphics, window, system modules)

## Build Steps

### Quick Build (Recommended)

Use the provided build script that handles temp directory setup automatically:

```bash
./build.sh
```

### Manual Build

If you need to build manually, first source the setup script to configure environment variables:

```bash
source setup.sh
```

Then follow these steps:

#### 1. Create the build directory

```bash
mkdir build
cd build
```

#### 2. Configure with CMake

```bash
cmake -G "MinGW Makefiles" -DSFML_DIR="C:/SFML-2.5.1/lib/cmake/SFML" ..
```

> If SFML is installed elsewhere, replace the path with your actual SFML CMake config directory.

#### 3. Build

```bash
cmake --build .
```

Or directly with Make:

```bash
mingw32-make
```

### Clean rebuild

```bash
cmake --build . --clean-first
```

Or remove the `build/` directory entirely and repeat from step 1.

## Output

All executables are placed in `build/bin/`:

| Executable          | Description                          |
|---------------------|--------------------------------------|
| `neural_selector`   | Main menu to select an algorithm     |
| `hebb_main`         | Hebb learning rule                   |
| `perceptron_main`   | Perceptron classifier                |
| `adaline_main`      | Adaline classifier                   |
| `madaline_main`     | Madaline classifier                  |
| `mlp_main`          | Multi-layer perceptron               |

Training data (`data/training.txt`) and SFML DLLs (on Windows) are automatically copied into `build/bin/` during the build.

## Troubleshooting

- **SFML not found**: Pass `-DSFML_DIR=<path>` pointing to the directory containing `SFMLConfig.cmake`.
- **Temp directory permission errors**: The `build.sh` script automatically handles this. If building manually, use `source setup.sh` first to configure temp directories.