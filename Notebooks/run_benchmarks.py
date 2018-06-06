#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import os
import timeit

# Constants
N_repeat = 10
N_loops = 100
N_loops_multiple = N_loops

BOX = np.array([100.,100.,100.,90,90,90],dtype=np.float32)
TRICLINIC_BOX = np.array([[100.0, 0, 0],[0, 100.0, 0],[0,0,100.0]], dtype=np.float32)
CUTOFF = 10
POS = BOX[:3]/2
NPOINTS = np.logspace(2, 5, num=10, dtype=np.int)


# Useful stuff
def pretty_time(value):

    multiplier = 0
    while value < 1:
        multiplier += 1
        value *= 1000

        if multiplier > 2:
            break

    units = ["s", "ms", "us", "ns"]
    return(value, units[multiplier])

# Benchmark runners
class BenchmarkRunnerSingleQuery(object):
    def __init__(self, name, filename):
        self.name = "{} (Single query)".format(name)
        self.filename = "benchmark-single_{}".format(filename)

        self.build_setup = ""
        self.build_stmt = ""

        self.query_setup = ""
        self.query_stmt = ""

        self.setup_skel = """
from __main__ import coords_dict, BOX, TRICLINIC_BOX, CUTOFF, POS
coords = coords_dict[{{}}]
{}
"""

    def set_setup_skel(self, setup_skel):
        self.setup_skel = setup_skel


    def set_build_benchmark(self, build_setup, build_stmt):
        self.build_setup = build_setup
        self.build_stmt = build_stmt


    def set_query_benchmark(self, query_setup, query_stmt):
        self.query_setup = query_setup
        self.query_stmt = query_stmt

    def run(self, repeat=N_repeat, number=N_loops):
        units = ["s", "ms", "us", "ns"]

        print("Benchmarking {}:".format(self.name))

        if self.build_stmt == "":
            print("NOTE: No build statement. Build benchmarking will be ignored")


        build_results = []
        query_results = []
        for ncoords in NPOINTS:

            # Timing build
            if self.build_stmt != "":

                timer_setup = self.setup_skel.format(self.build_setup).format(ncoords)

                build_times = timeit.repeat(
                    stmt=self.build_stmt,
                    setup=timer_setup,
                    repeat=repeat,
                    number=number
                )

                build_times = np.array(build_times) / number

                fastest = build_times.min()
                slowest = build_times.max()

                if slowest/fastest > 5:
                    warn_text = " Warning: slowest run is more then 5 times slower than fastest run."
                else:
                    warn_text = ""

                fastest = pretty_time(fastest)
                slowest = pretty_time(slowest)
                avg = pretty_time(build_times.mean())

                print("  - Build with {} coordinates: fastest run {:.1f} {} (average: {:.1f} {} - slowest: {:.1f} {}).{}".format(
                    ncoords,
                    fastest[0], fastest[1],
                    avg[0], avg[1],
                    slowest[0], slowest[1],
                    warn_text
                ))
            else:
                build_times = np.zeros((len(NPOINTS), repeat))


            # Timing query
            timer_setup = self.setup_skel.format(self.query_setup).format(ncoords)

            query_times = timeit.repeat(
                stmt=self.query_stmt,
                setup=timer_setup,
                repeat=repeat,
                number=number
            )

            query_times = np.array(query_times) / number

            fastest = query_times.min()
            slowest = query_times.max()

            if slowest / fastest > 5:
                warn_text = " Warning: slowest run is more then 5 times slower than fastest run."
            else:
                warn_text = ""

            fastest = pretty_time(fastest)
            slowest = pretty_time(slowest)
            avg = pretty_time(query_times.mean())

            print(
            "  - Single query with {} coordinates: fastest run {:.1f} {} (average: {:.1f} {} - slowest: {:.1f} {}).{}".format(
                ncoords,
                fastest[0], fastest[1],
                avg[0], avg[1],
                slowest[0], slowest[1],
                warn_text
            ))

            build_results.append(build_times)
            query_results.append(query_times)

        build_results = np.array(build_results)
        query_results = np.array(query_results)

        # Saving results
        np.savez(self.filename,
                 params=np.array([repeat, number]),
                 nparticles=NPOINTS,
                 build=build_results,
                 query=query_results)

        print("Done with {}. Benchmark results saved to '{}'\n".format(self.name, self.filename))


class BenchmarkRunnerMultipleQuery(object):
    def __init__(self, name, filename):
        self.name = "{} (Multiple queries)".format(name)
        self.filename = "benchmark-multiple_{}".format(filename)

        self.build_setup = ""
        self.build_stmt = ""

        self.query_setup = ""
        self.query_stmt = ""

        self.setup_skel = """
from __main__ import coords_dict, BOX, TRICLINIC_BOX, CUTOFF, POS
coords = coords_dict[{}]
{}
"""
        self.nqueries = np.logspace(0, 4, num=10, dtype=np.int)
        self.ncoords = 10000

    def set_setup_skel(self, setup_skel):
        self.setup_skel = setup_skel


    def set_build_benchmark(self, build_setup, build_stmt):
        self.build_setup = build_setup
        self.build_stmt = build_stmt


    def set_query_benchmark(self, query_setup, query_stmt):
        self.query_setup = query_setup
        self.query_stmt = query_stmt

    def run(self, repeat=N_repeat, number=N_loops_multiple):
        units = ["s", "ms", "us", "ns"]

        print("Benchmarking {}:".format(self.name))

        query_results = []
        for nquery in self.nqueries:

            # Timing query
            timer_setup = self.setup_skel.format(self.ncoords, self.query_setup).format(nquery)

            query_times = timeit.repeat(
                stmt=self.query_stmt,
                setup=timer_setup,
                repeat=repeat,
                number=number
            )

            query_times = np.array(query_times) / number

            fastest = query_times.min()
            slowest = query_times.max()

            if slowest / fastest > 5:
                warn_text = " Warning: slowest run is more then 5 times slower than fastest run."
            else:
                warn_text = ""

            fastest = pretty_time(fastest)
            slowest = pretty_time(slowest)
            avg = pretty_time(query_times.mean())

            print(
            "  - Multiple queries using {} queries: fastest run {:.1f} {} (average: {:.1f} {} - slowest: {:.1f} {}).{}".format(
                nquery,
                fastest[0], fastest[1],
                avg[0], avg[1],
                slowest[0], slowest[1],
                warn_text
            ))

            query_results.append(query_times)

        query_results = np.array(query_results)

        # Saving results
        np.savez(self.filename,
                 params=np.array([repeat, number, self.ncoords]),
                 nqueries=self.nqueries,
                 query=query_results)

        print("Done with {}. Benchmark results saved to '{}'\n".format(self.name, self.filename))


class BenchmarkRunnerParallelism(object):
    def __init__(self, name, filename):
        import multiprocessing

        self.name = "{} (Parallelism)".format(name)
        self.filename = "benchmark-parallelism_{}".format(filename)

        self.query_setup = ""
        self.query_stmt = ""

        self.setup_skel = """
from __main__ import coords_dict, BOX, TRICLINIC_BOX, CUTOFF, POS
coords = coords_dict[{{}}]
{}
"""
        self.ncoords = 10000
        self.nqueries = 1000
        self.nthreads = np.arange(1, multiprocessing.cpu_count() + 1)

    def set_setup_skel(self, setup_skel):
        self.setup_skel = setup_skel

    def set_query_benchmark(self, query_setup, query_stmt):
        self.query_setup = query_setup
        self.query_stmt = query_stmt

    def run(self, repeat=N_repeat, number=N_loops_multiple):
        units = ["s", "ms", "us", "ns"]

        print("Benchmarking {}:".format(self.name))

        query_results = []
        for n in self.nthreads:

            # Timing query
            timer_setup = self.setup_skel.format(self.query_setup).format(self.ncoords, self.nqueries, n)

            query_times = timeit.repeat(
                stmt=self.query_stmt,
                setup=timer_setup,
                repeat=repeat,
                number=number
            )

            query_times = np.array(query_times) / number

            fastest = query_times.min()
            slowest = query_times.max()

            if slowest / fastest > 5:
                warn_text = " Warning: slowest run is more then 5 times slower than fastest run."
            else:
                warn_text = ""

            fastest = pretty_time(fastest)
            slowest = pretty_time(slowest)
            avg = pretty_time(query_times.mean())

            print(
            "  - Single query using {} threads: fastest run {:.1f} {} (average: {:.1f} {} - slowest: {:.1f} {}).{}".format(
                n,
                fastest[0], fastest[1],
                avg[0], avg[1],
                slowest[0], slowest[1],
                warn_text
            ))

            query_results.append(query_times)

        query_results = np.array(query_results)

        # Saving results
        np.savez(self.filename,
                 params=np.array([repeat, number, self.nqueries, self.ncoords]),
                 nthreads=self.nthreads,
                 query=query_results)

        print("Done with {}. Benchmark results saved to '{}'\n".format(self.name, self.filename))


# Initialization of input data
coords_dict = defaultdict()
for ncoords in NPOINTS:
    fname = "benchmark_input_{}.txt".format(ncoords)
    if not os.path.isfile(fname):
        coords = (np.random.uniform(low=0, high=1.0, size=(ncoords, 3))*BOX[:3]).astype(np.float32)
        np.savetxt(fname, coords)
    else:
        coords = np.loadtxt(fname, dtype=np.float32)

    coords_dict[ncoords] = coords



if __name__ == "__main__":


    # Benchmarking MDAnalysis Periodic KDTree
    runner = BenchmarkRunnerSingleQuery("MDAnalysis Periodic KDTree", "mdakdtree")
    build_setup = "from MDAnalysis.lib.pkdtree import PeriodicKDTree"
    build_stmt = """
pkdt = PeriodicKDTree(BOX,bucket_size=10)
pkdt.set_coords(coords)
"""
    runner.set_build_benchmark(build_setup, build_stmt)
    query_setup = build_setup+build_stmt
    query_stmt = """
pkdt.search(POS,CUTOFF)
coords[pkdt.get_indices()]
"""
    runner.set_query_benchmark(query_setup, query_stmt)
    runner.run()


    # Benchmarking Biopython KDTree
    runner = BenchmarkRunnerSingleQuery("Biopython KDTree", "biokdtree")
    build_setup = "from Bio.KDTree import KDTree"
    build_stmt = """
kdtree = KDTree(dim=3)
kdtree.set_coords(coords)
"""
    runner.set_build_benchmark(build_setup, build_stmt)
    query_setup = build_setup + build_stmt
    query_stmt = """
kdtree.search(POS,CUTOFF)
coords[kdtree.get_indices()]
"""
    runner.set_query_benchmark(query_setup, query_stmt)
    runner.run()


    # Benchmarking Scipy cKDTree
    runner = BenchmarkRunnerSingleQuery("Scipy cKDTree", "ckdtree")
    build_setup = "import scipy.spatial as ss"
    build_stmt = """
kdtree = ss.cKDTree(coords)
"""
    runner.set_build_benchmark(build_setup, build_stmt)
    query_setup = build_setup + build_stmt
    query_stmt = """
coords[kdtree.query_ball_point(POS, CUTOFF)]
"""
    runner.set_query_benchmark(query_setup, query_stmt)
    runner.run()


    # Benchmarking Cython NS
    runner = BenchmarkRunnerSingleQuery("Cython NS", "cython")
    build_setup = """
from core_ns import FastNS
import numpy as np
pos=np.array([POS,])"""
    build_stmt = """
searcher = FastNS(TRICLINIC_BOX)
searcher.set_cutoff(CUTOFF)
searcher.set_coords(coords)
searcher.prepare()
"""
    runner.set_build_benchmark(build_setup, build_stmt)
    query_setup = build_setup + build_stmt
    query_stmt = """
searcher.search(pos)
"""
    runner.set_query_benchmark(query_setup, query_stmt)
    runner.run()


    # Benchmarking Scipy cKDTree (Multiple)
    runner = BenchmarkRunnerMultipleQuery("Scipy cKDTree", "ckdtree")
    build_setup = "import scipy.spatial as ss"
    build_stmt = """
kdtree = ss.cKDTree(coords)
pos=coords[:{}]
"""
    query_setup = build_setup + build_stmt
    query_stmt = """
kdtree.query_ball_point(pos, CUTOFF)
"""
    runner.set_query_benchmark(query_setup, query_stmt)
    runner.run()


    # Benchmarking Cython NS (Multiple query)
    runner = BenchmarkRunnerMultipleQuery("Cython NS", "cython")
    build_setup = """
from core_ns import FastNS
import numpy as np
pos=coords[:{}]"""
    build_stmt = """
searcher = FastNS(TRICLINIC_BOX)
searcher.set_cutoff(CUTOFF)
searcher.set_coords(coords)
searcher.set_nthreads(2, silent=True)
searcher.prepare()
"""
    query_setup = build_setup + build_stmt
    query_stmt = """
searcher.search(pos, return_ids=True)
"""
    runner.set_query_benchmark(query_setup, query_stmt)
    runner.run()


    # Benchmarking Cython NS (parallelism)
    runner = BenchmarkRunnerParallelism("Cython NS", "cython")
    build_setup = """
from core_ns import FastNS
import numpy as np
pos=coords[:{}]"""
    build_stmt = """
searcher = FastNS(TRICLINIC_BOX)
searcher.set_cutoff(CUTOFF)
searcher.set_nthreads({}, silent=True)
searcher.set_coords(coords)
searcher.prepare()
"""
    query_setup = build_setup + build_stmt
    query_stmt = """
searcher.search(pos, return_ids=True)
"""
    runner.set_query_benchmark(query_setup, query_stmt)
    runner.run()