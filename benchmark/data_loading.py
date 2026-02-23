################################################################################
# FILE: benchmarks/data_loading.py
################################################################################

"""Performance benchmarking script for data loading.

This script measures and compares data loading performance with different configurations
to identify bottlenecks and optimize the data pipeline.
"""

import sys
import os
import time
import logging
import yaml
from typing import Dict, List
import torch
from torch.utils.data import DataLoader

# Add parent directory to path to import src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import BiometricDataset, get_transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_dataloader(
    base_path: str,
    num_people: int,
    batch_size: int,
    num_workers: int,
    num_iterations: int = 50
) -> Dict[str, float]:
    """Benchmark data loading performance."""
    transform_fp, transform_iris = get_transforms(augment=False)
    dataset = BiometricDataset(base_path, num_people, transform_fp, transform_iris)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

    # Warmup
    logger.info(f"Warming up with num_workers={num_workers}, batch_size={batch_size}...")
    for i, batch in enumerate(loader):
        if i >= 2:
            break

    # Benchmark
    logger.info(f"Benchmarking {num_iterations} iterations...")
    times: List[float] = []
    start_total = time.time()

    for i, batch in enumerate(loader):
        if i >= num_iterations:
            break
        batch_start = time.time()
        
        # Simulate minimal processing (just move to device if available)
        if torch.cuda.is_available():
            _ = batch[0].cuda(non_blocking=True)
            
        batch_time = time.time() - batch_start
        times.append(batch_time)

    total_time = time.time() - start_total

    return {
        'num_workers': num_workers,
        'batch_size': batch_size,
        'total_time': total_time,
        'avg_batch_time': sum(times) / len(times),
        'min_batch_time': min(times),
        'max_batch_time': max(times),
        'samples_per_sec': (num_iterations * batch_size) / total_time,
        'batches_per_sec': num_iterations / total_time
    }


def run_benchmarks(base_path: str, num_people: int) -> None:
    """Run comprehensive benchmarks with different configurations."""
    logger.info("=" * 80)
    logger.info("DATA LOADING PERFORMANCE BENCHMARK")
    logger.info("=" * 80)

    # Test configurations
    worker_configs = [0, 2, 4, 8]
    batch_sizes = [4, 8, 16]

    results = []

    for batch_size in batch_sizes:
        for num_workers in worker_configs:
            logger.info(f"\nTesting: batch_size={batch_size}, num_workers={num_workers}")
            try:
                result = benchmark_dataloader(
                    base_path, num_people, batch_size, num_workers, num_iterations=20
                )
                results.append(result)

                logger.info(
                    f"  Total time: {result['total_time']:.2f}s | "
                    f"Avg batch: {result['avg_batch_time']:.4f}s | "
                    f"Throughput: {result['samples_per_sec']:.1f} samples/s"
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - Best Configurations")
    logger.info("=" * 80)

    if not results:
        logger.warning("No results to summarize.")
        return

    # Find best by throughput
    best = max(results, key=lambda x: x['samples_per_sec'])
    logger.info(
        f"\nBest Throughput: {best['samples_per_sec']:.1f} samples/s\n"
        f"  Config: batch_size={best['batch_size']}, num_workers={best['num_workers']}\n"
        f"  Avg batch time: {best['avg_batch_time']:.4f}s"
    )

    # Print comparison table
    logger.info("\nDetailed Results:")
    logger.info(f"{'Batch':<8}{'Workers':<10}{'Throughput':<15}{'Avg Time':<12}")
    logger.info("-" * 45)
    for r in sorted(results, key=lambda x: x['samples_per_sec'], reverse=True):
        logger.info(
            f"{r['batch_size']:<8}{r['num_workers']:<10}"
            f"{r['samples_per_sec']:>10.1f} s/s   {r['avg_batch_time']:>8.4f}s"
        )


def main():
    """Main benchmarking function."""
    # Load config - path relative to project root
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.yaml')
    
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        return

    logger.info(f"Dataset path: {cfg['data_path']}")
    logger.info(f"Number of people: {cfg['num_people']}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CPU count: {torch.multiprocessing.cpu_count()}")

    run_benchmarks(cfg['data_path'], cfg['num_people'])


if __name__ == '__main__':
    main()