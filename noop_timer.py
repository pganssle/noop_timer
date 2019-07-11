import timeit
import operator

import io
import json
import os
import zipfile

from codetransformer import Code
from codetransformer import instructions as inst

import numpy as np

DEFAULT_N = 10000
DEFAULT_k = 5
DEFAULT_K = 25
DEFAULT_RANGE = (0, 150, 1)

DEFAULT_PLOT_LOC = "noop_timings.png"
DEFAULT_DATA_LOC = "noop_data.zip"


def _time_func(f, N, k=3, msg=True):
    """Basic version of IPython's %timeit magic"""
    t = min(timeit.timeit(f, number=N) for _ in range(k))

    t_per = t / N
    if msg:
        if t_per < 1e-6:
            units = 'ns'
            t_per *= 1e9
        elif t_per < 1e-3:
            units = 'µs'
            t_per *= 1e6
        else:
            units = 'ms'
            t_per *= 1e3

        print(f'{N} loops, best of {k}: {t_per:0.4} {units}')
    return t_per


def make_noop_func(n):
    """Create a function with N noops"""
    co = Code([inst.NOP() for _ in range(n)] +
              [inst.LOAD_CONST(None), inst.RETURN_VALUE()])
    f = lambda: None
    f.__code__ = co.to_pycode()

    return f


class Runner:
    def __init__(self, N=DEFAULT_N, k=DEFAULT_k, K=DEFAULT_K,
                 noop_range=DEFAULT_RANGE,
                 verbose=True, progress=True):
        self._print = print if verbose else lambda x: None

        if progress:
            import progressbar
            self._pbar = progressbar.progressbar
        else:
            self._pbar = lambda x: x

        self.N = N
        self.k = k
        self.K = K
        self.noop_range = noop_range

        self.n_vals = None
        self.times_per = None
        self._time_per = None
        self._has_run = False

    def run(self):
        """
        Run the timing function.

        This attempts to accurately determine how long it takes to run
        functions consisting of only *i* ``NOP`` operations, where *i* is all
        the values of ``range(*noop_range)``.

        It uses ``timeit`` with ``number=N`` and repeats this ``k`` times,
        taking the fastest time as the correct value (biased towards the CPU
        operating optimally). This measurement is repreated ``K`` times.
        """
        N = self.N
        k = self.k
        K = self.K

        nvals = list(range(*self.noop_range))
        times_per = np.zeros((len(nvals), K))

        self._print('Starting timing')
        for jj in self._pbar(range(K)):
            eval_order = list(range(len(nvals)))
            np.random.shuffle(eval_order)

            for ii in eval_order:
                n = nvals[ii]
                t_per = _time_func(make_noop_func(n), N, k, msg=False)

                times_per[ii, jj] = t_per * 1e9

        self.times_per = times_per
        self.n_vals = np.atleast_2d(np.asarray(nvals)).T
        self._has_run = True

    def _verify_run(self):
        if self._has_run:
            return

        raise RunnerNotInitializedError("Cannot access run data before " +
                                        "running the Runner.run function")

    def write_data(self, fname=DEFAULT_DATA_LOC):
        self._verify_run()
        import cpuinfo

        X_agg = np.hstack((self.n_vals, self.time_per))
        X_all = np.hstack((self.n_vals, self.times_per))

        agg_header = ["num_nops", "time_ns"]
        all_header = ["num_nops"] + list(map(str,
                                             range(0, self.times_per.shape[1])))
        delimiter = ","

        with zipfile.ZipFile(fname, mode='w') as zf:
            self._print("Writing data")
            # Write the "aggregate" file
            agg_f = io.StringIO()
            header = delimiter.join(agg_header)
            np.savetxt(agg_f, X_agg, delimiter=delimiter, header=header)

            agg_f.seek(0)
            zf.writestr("agg_data.csv", data=agg_f.read())

            # Write the "all" file
            all_f = io.StringIO()
            header = delimiter.join(all_header)
            np.savetxt(all_f, X_all, delimiter=delimiter, header=header)

            all_f.seek(0)
            zf.writestr("all_data.csv", data=all_f.read())

            # Write the run configuration
            run_config = {
                "N": self.N,
                "k": self.k,
                "K": self.K,
                "noop_range": self.noop_range,
            }

            zf.writestr("run_config.json", data=json.dumps(run_config))

            # Write the CPU information
            self._print("Writing CPU information")
            zf.writestr("cpu_info.json", data=cpuinfo.get_cpu_info_json())

            # Write other system information
            self._print("Writing system configuration")
            zf.writestr("sys_config.json", data=self.get_system_info())

    def plot_timing(self, fname=DEFAULT_PLOT_LOC):
        self._verify_run()
        import matplotlib
        from matplotlib import pyplot as plt
        self._print('Plotting')

        plt.plot(self.n_vals, self.time_per)
        f = plt.gcf()

        plt.title(f'Average of {self.K} runs, timing {self.N} ' +
                  f'calls (best of {self.k})')
        plt.xlabel('Number of NOP opcodes')
        plt.ylabel('Time (ns)')

        plt.tight_layout()

        f.savefig(fname)
        plt.show()

    def get_system_info(self):
        import sysconfig

        configs = dict(sysconfig.get_config_vars())

        return json.dumps(configs)

    @property
    def time_per(self):
        self._verify_run()
        if self._time_per is None:
            self._time_per = np.atleast_2d(self.times_per.mean(axis=1)).T

        return self._time_per


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-N", "--num-calls", type=int,
                        default=DEFAULT_N,
                        help="Number of times to run the function per " +
                              "timing (timeit's 'number')")
    parser.add_argument("-k", "--num-timings", type=int,
                        default=DEFAULT_k,
                        help="Number of times to time the function (minimum " +
                             "is taken)")
    parser.add_argument("-K", "--num-averages", type=int,
                        default=DEFAULT_K,
                        help="Number of times to run the full process (best " +
                             "of k with N timings) for all numbers of noops " +
                             "- to be averaged")
    parser.add_argument("--noop-start", type=int, default=DEFAULT_RANGE[0],
                        help="Minimum number of noops to run")
    parser.add_argument("--noop-stop", type=int, default=DEFAULT_RANGE[1],
                        help="Maximum number of noops to time")
    parser.add_argument("--noop-step", type=int, default=DEFAULT_RANGE[2],
                        help="Number of noops per step (e.g. 2 would test "
                             "0, 2, 4, 6)")
    parser.add_argument("-d", "--data-out", default=DEFAULT_DATA_LOC,
                        help="Location to write the timing data (zip file)")
    parser.add_argument("--no-write-data", action="store_true",
                        help="Disable writing output data")
    parser.add_argument("-p", "--plot-out", default=DEFAULT_PLOT_LOC,
                        help="Location to write the plot file")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting of output values")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the progress bar")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print to stdout")

    args = parser.parse_args()
    N = args.num_calls
    k = args.num_timings
    K = args.num_averages

    noop_range = (args.noop_start, args.noop_stop, args.noop_step)

    runner = Runner(N=N, k=k, K=K, noop_range=noop_range,
                    progress=(not args.no_progress), verbose=(not args.quiet))

    runner.run()

    if not args.no_write_data:
        runner.write_data(args.data_out)

    if not args.no_plot:
        runner.plot_timing(args.plot_out)


class RunnerNotInitializedError(Exception):
    """Exception raised when trying to access Runner data before it is generated"""

if __name__ == "__main__":
    main()
