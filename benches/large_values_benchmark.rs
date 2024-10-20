use std::env::current_dir;
use tempfile::{NamedTempFile, TempDir};

mod common;
use common::*;

use rand::Rng;
use std::time::{Duration, Instant};

const ELEMENTS: usize = 1_000_000;

/// Returns pairs of key, value
fn gen_data(count: usize, key_size: usize, value_size: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut pairs = vec![];

    for _ in 0..count {
        let key: Vec<u8> = (0..key_size).map(|_| rand::thread_rng().gen()).collect();
        let value: Vec<u8> = (0..value_size).map(|_| rand::thread_rng().gen()).collect();
        pairs.push((key, value));
    }

    pairs
}

fn benchmark<T: BenchDatabase>(db: T) -> Vec<(&'static str, Duration)> {
    let mut results = Vec::new();
    let mut pairs = gen_data(1_000_000, 24, 150);
    let mut written = 0;

    let mut bigpairs = gen_data(100, 24, 2_000_000);
    let bigelements = 4000;

    let start = Instant::now();
    let mut txn = db.write_transaction();
    let mut inserter = txn.get_inserter();
    {
        for _ in 0..bigelements {
            let len = bigpairs.len();
            let (key, value) = &mut bigpairs[written % len];
            key[16..].copy_from_slice(&(written as u64).to_le_bytes());
            inserter.insert(key, value).unwrap();
            written += 1;
        }
        for _ in 0..ELEMENTS {
            let len = pairs.len();
            let (key, value) = &mut pairs[written % len];
            key[16..].copy_from_slice(&(written as u64).to_le_bytes());
            inserter.insert(key, value).unwrap();
            written += 1;
        }
    }
    drop(inserter);
    txn.commit().unwrap();

    let end = Instant::now();
    let duration = end - start;
    println!(
        "{}: Bulk loaded {} 2MB items and {} small items in {}ms",
        T::db_type_name(),
        bigelements,
        ELEMENTS,
        duration.as_millis()
    );
    results.push(("bulk load (2MB values)", duration));

    results
}

fn main() {
    let redb_latency_results = {
        let tmpfile: NamedTempFile = NamedTempFile::new_in(current_dir().unwrap()).unwrap();
        let db = redb::Database::builder().create(tmpfile.path()).unwrap();
        let table = RedbBenchDatabase::new(&db);
        benchmark(table)
    };

    let lmdb_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(current_dir().unwrap()).unwrap();
        let env = unsafe {
            heed::EnvOpenOptions::new()
                .map_size(10 * 4096 * 1024 * 1024)
                .open(tmpfile.path())
                .unwrap()
        };
        let table = HeedBenchDatabase::new(&env);
        benchmark(table)
    };

    let canopydb_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(current_dir().unwrap()).unwrap();
        let mut env_opts = canopydb::EnvOptions::new(tmpfile.path());
        env_opts.page_cache_size = 4 * 1024 * 1024 * 1024;
        let db = canopydb::Database::with_options(env_opts, Default::default()).unwrap();
        let db_bench = CanopydbBenchDatabase::new(&db);
        benchmark(db_bench)
    };

    let rocksdb_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(current_dir().unwrap()).unwrap();

        let mut bb = rocksdb::BlockBasedOptions::default();
        bb.set_block_cache(&rocksdb::Cache::new_lru_cache(4 * 1_024 * 1_024 * 1_024));

        let mut opts = rocksdb::Options::default();
        opts.set_block_based_table_factory(&bb);
        opts.create_if_missing(true);

        let db = rocksdb::TransactionDB::open(&opts, &Default::default(), tmpfile.path()).unwrap();
        let table = RocksdbBenchDatabase::new(&db);
        benchmark(table)
    };

    let sled_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(current_dir().unwrap()).unwrap();
        let db = sled::Config::new().path(tmpfile.path()).open().unwrap();
        let table = SledBenchDatabase::new(&db, tmpfile.path());
        benchmark(table)
    };

    let mut rows = Vec::new();

    for (benchmark, _duration) in &redb_latency_results {
        rows.push(vec![benchmark.to_string()]);
    }

    for results in [
        redb_latency_results,
        canopydb_results,
        sled_results,
        lmdb_results,
        rocksdb_results,
    ] {
        for (i, (_benchmark, duration)) in results.iter().enumerate() {
            rows[i].push(format!("{}ms", duration.as_millis()));
        }
    }

    let mut table = comfy_table::Table::new();
    table.set_width(100);
    table.set_header(["", "redb", "canopydb", "sled", "lmdb", "rocksdb"]);
    for row in rows {
        table.add_row(row);
    }

    println!();
    println!("{table}");
}
