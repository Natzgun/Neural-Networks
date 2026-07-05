# Neural Network

## Compile

### CNN

```bash
cmake -B build -S .
cmake --build build -j$(nproc)

# To execute with datasets [simpsons, mnist, bloodmnist]
./build/neural_networks simpsons

# to format the code
clang-format -i $(git ls-files '*.cpp' '*.hpp' '*.cu' '*.cuh')
```
