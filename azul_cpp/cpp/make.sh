# emcc wasm_binding.cpp -O3 \
#     -s WASM=1 \
#     -s ALLOW_MEMORY_GROWTH=1 \
#     -s MODULARIZE=1 \
#     -s EXPORT_ES6=1 \
#     -s "EXPORT_NAME='AzulModule'" \
#     -pthread \
#     -s USE_PTHREADS=1 \
#     -s PTHREAD_POOL_SIZE=4 \
#     -s PROXY_TO_PTHREAD=1 \
#     --bind \
#     -o azul.js

#!/bin/bash

# 1. 设置输出文件名
emcc wasm_binding.cpp -O0 \
    -g3 \
    -s ASSERTIONS=2 \
    -s STACK_OVERFLOW_CHECK=2 \
    -s SAFE_HEAP=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s "EXPORT_NAME='AzulModule'" \
    --bind \
    -o ../web/azul.js