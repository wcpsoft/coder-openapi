#!/bin/bash

# 确保flamegraph工具已安装
cargo install flamegraph

# 运行性能分析
cargo flamegraph --bin coder-openapi -- --bench

# 生成性能报告
echo "Performance analysis completed. Flamegraph saved to flamegraph.svg"
