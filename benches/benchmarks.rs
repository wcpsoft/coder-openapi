//! 性能基准测试
//!
//! 本模块包含项目的性能基准测试

use coder_openapi::service::models::ModelManager;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn model_manager_benchmark(c: &mut Criterion) {
    c.bench_function("model_manager_init", |b| {
        b.iter(|| {
            let manager = ModelManager::new();
            black_box(manager);
        });
    });
}

criterion_group!(benches, model_manager_benchmark);
criterion_main!(benches);
