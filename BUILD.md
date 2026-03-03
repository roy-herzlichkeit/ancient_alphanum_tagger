# Build Instructions

## Prerequisites

- **CMake** >= 3.10
- **MinGW** (GCC) with C++11 support
- **SFML** >= 2.5 (graphics, window, system modules)

## Build Steps

### 1. Create the build directory

```bash
mkdir build
cd build
```

### 2. Configure with CMake

```bash
cmake -G "MinGW Makefiles" -DSFML_DIR="C:/SFML-2.5.1/lib/cmake/SFML" ..
```

> If SFML is installed elsewhere, replace the path with your actual SFML CMake config directory.

### 3. Build

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
- **Temp directory permission errors**: Set `TEMP` and `TMP` environment variables to a writable directory before running CMake:
  ```bash
  export TEMP="$HOME/AppData/Local/Temp"
  export TMP="$TEMP"
  ```
### Extra notes:

    I'm using:
    ```bash
    cd "E:/Projects/_PSC_files_/ALPHANUM_DETECTOR" && rm -rf build && mkdir build && cd build && export TEMP="$HOME/AppData/Local/Temp" && export TMP="$TEMP"
    ```
    to set the temp of this folder as my original one doesn't permit to do anything and i mistakenly deleted some of my sys paths which I still can't fix