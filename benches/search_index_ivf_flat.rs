use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faiss::{
    index::{flat::FlatIndexImpl, ivf_flat::IVFFlatIndexImpl},
    ConcurrentIndex, Index,
};
use rand::Rng;

pub fn generate_random_vectors(dimension: usize, size: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut data = vec![0f32; dimension * size];
    for slot in data.iter_mut() {
        *slot = rng.random();
    }
    data
}

pub fn build_index(
    dimension: u32,
    nlist: u32,
    train_size: usize,
    dataset_size: usize,
) -> faiss::error::Result<IVFFlatIndexImpl> {
    let q = FlatIndexImpl::new_l2(dimension).unwrap();
    let mut index = IVFFlatIndexImpl::new_l2(q, dimension, nlist).unwrap();

    let train_vectors = generate_random_vectors(dimension as usize, train_size);
    eprintln!("training index...");
    index.train(&train_vectors)?;

    let dataset = generate_random_vectors(dimension as usize, dataset_size);
    eprintln!("adding dataset to index...");
    index.add(&dataset)?;

    Ok(index)
}

fn search_ivf_flat(c: &mut Criterion) {
    const DIMENSION: u32 = 256;
    const NLIST: u32 = 10;
    const TRAIN_SIZE: usize = 1_000;
    const DATASET_SIZE: usize = 1_000_000;
    const NEIGHBORS: usize = 25;

    let index = build_index(DIMENSION, NLIST, TRAIN_SIZE, DATASET_SIZE).unwrap();
    eprintln!("finished building index");

    let total_steps = 11usize;
    let sizes: Vec<_> = (0..total_steps).into_iter().map(|v| 1 << v).collect();
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(sizes.len());
    for size in sizes.iter() {
        queries.push(generate_random_vectors(DIMENSION as usize, *size));
    }

    let mut group = c.benchmark_group("search_ivf_flat");
    for (idx, size) in sizes.iter().enumerate() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                index.search(&queries[idx], NEIGHBORS).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, search_ivf_flat,);
criterion_main!(benches);
