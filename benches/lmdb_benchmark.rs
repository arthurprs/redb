use std::env::current_dir;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::{fs, process, thread};
use tempfile::{NamedTempFile, TempDir};

mod common;
use common::*;

use std::time::{Duration, Instant};

const ITERATIONS: usize = 2;
const BULK_ELEMENTS: usize = 1_000_000;
const KEY_SIZE: usize = 24;
const VALUE_SIZE: usize = 150;
const RNG_SEED: u64 = 3;

fn fill_slice(slice: &mut [u8], rng: &mut fastrand::Rng) {
    rng.fill(slice);
}

/// Returns pairs of key, value
fn gen_pair(rng: &mut fastrand::Rng) -> ([u8; KEY_SIZE], Vec<u8>) {
    let mut key = [0u8; KEY_SIZE];
    fill_slice(&mut key, rng);
    let mut value = vec![0u8; VALUE_SIZE];
    fill_slice(&mut value, rng);

    (key, value)
}

fn make_rng() -> fastrand::Rng {
    fastrand::Rng::with_seed(RNG_SEED)
}

fn make_rng_shards(shards: usize, elements: usize) -> Vec<fastrand::Rng> {
    let mut rngs = vec![];
    let elements_per_shard = elements / shards;
    for i in 0..shards {
        let mut rng = make_rng();
        for _ in 0..(i * elements_per_shard) {
            gen_pair(&mut rng);
        }
        rngs.push(rng);
    }

    rngs
}

fn benchmark<T: BenchDatabase + Send + Sync>(db: T) -> Vec<(String, ResultType)> {
    // Throttle down the cpu, os, fs and the nvme for a bit if configured.
    // Do it on the top of the function so it's _after_ the file/folder removal
    if let Ok(sleep_str) = std::env::var("SLEEP_SECS_BETWEEN") {
        std::thread::sleep(Duration::from_secs(sleep_str.parse().unwrap()));
    }

    let mut rng = make_rng();
    let mut results = Vec::new();
    let db = Arc::new(db);

    let start = Instant::now();
    let mut txn = db.write_transaction();
    let mut inserter = txn.get_inserter();
    {
        for _ in 0..BULK_ELEMENTS {
            let (key, value) = gen_pair(&mut rng);
            inserter.insert(&key, &value).unwrap();
        }
    }
    drop(inserter);
    txn.commit().unwrap();

    let end = Instant::now();
    let duration = end - start;
    println!(
        "{}: Bulk loaded {} items in {}ms",
        T::db_type_name(),
        BULK_ELEMENTS,
        duration.as_millis()
    );
    results.push(("bulk load".to_string(), ResultType::Duration(duration)));

    let start = Instant::now();
    let individual_writes = 1000;
    {
        for _ in 0..individual_writes {
            let mut txn = db.write_transaction();
            let mut inserter = txn.get_inserter();
            let (key, value) = gen_pair(&mut rng);
            inserter.insert(&key, &value).unwrap();
            drop(inserter);
            txn.commit().unwrap();
        }
    }

    let end = Instant::now();
    let duration = end - start;
    println!(
        "{}: Wrote {} individual items in {}ms",
        T::db_type_name(),
        individual_writes,
        duration.as_millis()
    );
    results.push((
        "individual writes".to_string(),
        ResultType::Duration(duration),
    ));

    let start = Instant::now();
    let batch_writes = 100;
    let batch_size = 1000;
    {
        for _ in 0..batch_writes {
            let mut txn = db.write_transaction();
            let mut inserter = txn.get_inserter();
            for _ in 0..batch_size {
                let (key, value) = gen_pair(&mut rng);
                inserter.insert(&key, &value).unwrap();
            }
            drop(inserter);
            txn.commit().unwrap();
        }
    }

    let end = Instant::now();
    let duration = end - start;
    println!(
        "{}: Wrote {} x {} items in {}ms",
        T::db_type_name(),
        batch_writes,
        batch_size,
        duration.as_millis()
    );
    results.push(("batch writes".to_string(), ResultType::Duration(duration)));

    let elements = BULK_ELEMENTS + individual_writes + batch_size * batch_writes;
    let txn = db.read_transaction();
    {
        {
            let start = Instant::now();
            let len = txn.get_reader().len();
            assert_eq!(len, elements as u64);
            let end = Instant::now();
            let duration = end - start;
            println!("{}: len() in {}ms", T::db_type_name(), duration.as_millis());
            results.push(("len()".to_string(), ResultType::Duration(duration)));
        }

        for _ in 0..ITERATIONS {
            let mut rng = make_rng();
            let start = Instant::now();
            let mut checksum = 0u64;
            let mut expected_checksum = 0u64;
            let reader = txn.get_reader();
            for _ in 0..elements {
                let (key, value) = gen_pair(&mut rng);
                let result = reader.get(&key).unwrap();
                checksum += result.as_ref()[0] as u64;
                expected_checksum += value[0] as u64;
            }
            assert_eq!(checksum, expected_checksum);
            let end = Instant::now();
            let duration = end - start;
            println!(
                "{}: Random read {} items in {}ms",
                T::db_type_name(),
                elements,
                duration.as_millis()
            );
            results.push(("random reads".to_string(), ResultType::Duration(duration)));
        }

        for _ in 0..ITERATIONS {
            let mut rng = make_rng();
            let start = Instant::now();
            let reader = txn.get_reader();
            let mut value_sum = 0;
            let num_scan = 10;
            for _ in 0..elements {
                let (key, _value) = gen_pair(&mut rng);
                let mut iter = reader.range_from(&key);
                for _ in 0..num_scan {
                    if let Some((_, value)) = iter.next() {
                        value_sum += value.as_ref()[0];
                    } else {
                        break;
                    }
                }
            }
            assert!(value_sum > 0);
            let end = Instant::now();
            let duration = end - start;
            println!(
                "{}: Random range read {} elements in {}ms",
                T::db_type_name(),
                elements * num_scan,
                duration.as_millis()
            );
            results.push((
                "random range reads".to_string(),
                ResultType::Duration(duration),
            ));
        }
    }
    drop(txn);

    for num_threads in [4, 8, 16, 32] {
        let barrier = Arc::new(std::sync::Barrier::new(num_threads));
        let mut rngs = make_rng_shards(num_threads, elements);
        let start = Instant::now();

        thread::scope(|s| {
            for _ in 0..num_threads {
                let barrier = barrier.clone();
                let db2 = db.clone();
                let rng = rngs.pop().unwrap();
                s.spawn(move || {
                    barrier.wait();
                    for _ in 0..ITERATIONS {
                        let txn = db2.read_transaction();
                        let mut checksum = 0u64;
                        let mut expected_checksum = 0u64;
                        let reader = txn.get_reader();
                        let mut rng = rng.clone();
                        for _ in 0..(elements / num_threads) {
                            let (key, value) = gen_pair(&mut rng);
                            let result = reader.get(&key).unwrap();
                            checksum += result.as_ref()[0] as u64;
                            expected_checksum += value[0] as u64;
                        }
                        assert_eq!(checksum, expected_checksum);
                    }
                });
            }
        });

        let end = Instant::now();
        let duration = end - start;
        println!(
            "{}: Random read ({} threads) {} items in {}ms",
            T::db_type_name(),
            num_threads,
            elements,
            duration.as_millis()
        );
        results.push((
            format!("random reads ({num_threads} threads)"),
            ResultType::Duration(duration),
        ));
    }

    let start = Instant::now();
    let deletes = elements / 2;
    {
        let mut rng = make_rng();
        let mut txn = db.write_transaction();
        let mut inserter = txn.get_inserter();
        for _ in 0..deletes {
            let (key, _value) = gen_pair(&mut rng);
            inserter.remove(&key).unwrap();
        }
        drop(inserter);
        txn.commit().unwrap();
    }

    let end = Instant::now();
    let duration = end - start;
    println!(
        "{}: Removed {} items in {}ms",
        T::db_type_name(),
        deletes,
        duration.as_millis()
    );
    results.push(("bulk removals".to_string(), ResultType::Duration(duration)));

    results
}

fn database_size(path: &Path) -> u64 {
    let mut size = 0u64;
    for result in walkdir::WalkDir::new(path) {
        let entry = result.unwrap();
        size += entry.metadata().unwrap().len();
    }
    size
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ResultType {
    Duration(Duration),
    SizeInBytes(u64),
    // Have NA last so it sorts last
    NA,
}

impl std::fmt::Display for ResultType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use byte_unit::{Byte, UnitType};

        match self {
            ResultType::NA => write!(f, "N/A"),
            ResultType::Duration(d) => write!(f, "{d:.2?}"),
            ResultType::SizeInBytes(s) => {
                let b = Byte::from_u64(*s).get_appropriate_unit(UnitType::Binary);
                write!(f, "{b:.2}")
            }
        }
    }
}

fn main() {
    let _ = env_logger::try_init();
    let tmpdir = current_dir().unwrap().join(".benchmark");
    fs::create_dir(&tmpdir).unwrap();

    let tmpdir2 = tmpdir.clone();
    ctrlc::set_handler(move || {
        fs::remove_dir_all(&tmpdir2).unwrap();
        process::exit(1);
    })
    .unwrap();

    let redb_latency_results = {
        let tmpfile: NamedTempFile = NamedTempFile::new_in(&tmpdir).unwrap();
        let mut db = redb::Database::builder()
            .set_cache_size(4 * 1024 * 1024 * 1024)
            .create(tmpfile.path())
            .unwrap();
        let table = RedbBenchDatabase::new(&db);
        let mut results = benchmark(table);

        let size = database_size(tmpfile.path());
        results.push((
            "size pre-compact".to_string(),
            ResultType::SizeInBytes(size),
        ));

        let start = Instant::now();
        db.compact().unwrap();
        let end = Instant::now();
        let duration = end - start;
        println!("redb: Compacted in {}ms", duration.as_millis());
        results.push(("compaction".to_string(), ResultType::Duration(duration)));

        let size = database_size(tmpfile.path());
        results.push((
            "size after bench".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results
    };

    let lmdb_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();
        let env = unsafe {
            heed::EnvOpenOptions::new()
                .map_size(4096 * 1024 * 1024)
                .open(tmpfile.path())
                .unwrap()
        };
        let table = HeedBenchDatabase::new(&env);
        let mut results = benchmark(table);
        let size = database_size(tmpfile.path());
        results.push((
            "size pre-compact".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results.push(("compaction".to_string(), ResultType::NA));
        results.push((
            "size after bench".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results
    };

    let canopydb_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();
        let mut env_opts = canopydb::EnvOptions::new(tmpfile.path());
        env_opts.page_cache_size = 4 * 1024 * 1024 * 1024;
        let db = canopydb::Database::with_options(env_opts, Default::default()).unwrap();
        let db_bench = CanopydbBenchDatabase::new(&db);
        let mut results = benchmark(db_bench);

        let size = database_size(tmpfile.path());
        results.push((
            "size pre-compact".to_string(),
            ResultType::SizeInBytes(size),
        ));
        let start = Instant::now();
        db.compact().unwrap();
        let end = Instant::now();
        let duration = end - start;
        println!("canopydb: Compacted in {}ms", duration.as_millis());
        results.push(("compaction".to_string(), ResultType::Duration(duration)));

        // wait a couple seconds to observe the size after a wal cleanup
        std::thread::sleep(Duration::from_secs(2));
        let size = database_size(tmpfile.path());
        results.push((
            "size after bench".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results
    };

    let rocksdb_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();

        let mut bb = rocksdb::BlockBasedOptions::default();
        bb.set_block_cache(&rocksdb::Cache::new_lru_cache(4 * 1_024 * 1_024 * 1_024));

        let mut opts = rocksdb::Options::default();
        opts.set_block_based_table_factory(&bb);
        opts.create_if_missing(true);

        let db = rocksdb::TransactionDB::open(&opts, &Default::default(), tmpfile.path()).unwrap();
        let table = RocksdbBenchDatabase::new(&db);
        let mut results = benchmark(table);
        let size = database_size(tmpfile.path());
        results.push((
            "size pre-compact".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results.push(("compaction".to_string(), ResultType::NA));
        results.push((
            "size after bench".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results
    };

    let sled_results = {
        let tmpfile: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();
        let db = sled::Config::new().path(tmpfile.path()).open().unwrap();
        let table = SledBenchDatabase::new(&db, tmpfile.path());
        let mut results = benchmark(table);
        let size = database_size(tmpfile.path());
        results.push((
            "size pre-compact".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results.push(("compaction".to_string(), ResultType::NA));
        results.push((
            "size after bench".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results
    };

    let sanakirja_results = {
        let tmpfile: NamedTempFile = NamedTempFile::new_in(&tmpdir).unwrap();
        fs::remove_file(tmpfile.path()).unwrap();
        let db = sanakirja::Env::new(tmpfile.path(), 4096 * 1024, 2).unwrap();
        let table = SanakirjaBenchDatabase::new(&db);
        let mut results = benchmark(table);
        let size = database_size(tmpfile.path());
        results.push((
            "size pre-compact".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results.push(("compaction".to_string(), ResultType::NA));
        results.push((
            "size after bench".to_string(),
            ResultType::SizeInBytes(size),
        ));
        results
    };

    fs::remove_dir_all(&tmpdir).unwrap();

    let mut rows = Vec::new();

    for (benchmark, _duration) in &redb_latency_results {
        rows.push(vec![benchmark.to_string()]);
    }

    let results = [
        (redb_latency_results, true),
        (canopydb_results, true),
        (sled_results, true),
        (sanakirja_results, true),
        (lmdb_results, false),
        (rocksdb_results, false),
    ];

    let mut identified_smallests = vec![vec![false; results.len()]; rows.len()];
    let mut identified_smallests_rust_only = vec![vec![false; results.len()]; rows.len()];
    for (i, (identified_smallests_row, identified_smallests_rust_only_row)) in identified_smallests
        .iter_mut()
        .zip(&mut identified_smallests_rust_only)
        .enumerate()
    {
        let mut smallest = None;
        let mut smallest_rust_only = None;
        for j in 0..identified_smallests_row.len() {
            let rust_only = results[j].1;
            let (_, rt) = &results[j].0[i];
            smallest = match smallest {
                Some((_, prev)) if rt < prev => Some((j, rt)),
                Some((pi, prev)) => Some((pi, prev)),
                None => Some((j, rt)),
            };
            if rust_only {
                smallest_rust_only = match smallest_rust_only {
                    Some((_, prev)) if rt < prev => Some((j, rt)),
                    Some((pi, prev)) => Some((pi, prev)),
                    None => Some((j, rt)),
                };
            }
        }
        let (j, _rt) = smallest.unwrap();
        identified_smallests_row[j] = true;
        let (j, _rt) = smallest_rust_only.unwrap();
        identified_smallests_rust_only_row[j] = true;
    }

    for (j, (results, _)) in results.iter().enumerate() {
        for (i, (_benchmark, result_type)) in results.iter().enumerate() {
            let mut modifier = String::new();
            if identified_smallests[i][j] {
                modifier += "**";
            }
            if identified_smallests_rust_only[i][j] {
                modifier += "*";
            }
            rows[i].push(format!("{modifier}{result_type}{modifier}"));
        }
    }

    let mut table = comfy_table::Table::new();
    table.load_preset(comfy_table::presets::ASCII_MARKDOWN);
    table.set_width(100);
    table.set_header([
        "",
        "redb",
        "canopydb",
        "sled",
        "sanakirja",
        "lmdb",
        "rocksdb",
    ]);
    for row in rows {
        table.add_row(row);
    }

    println!();
    println!("{table}");
}
