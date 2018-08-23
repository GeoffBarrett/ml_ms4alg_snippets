from mountainlab_pytools import mdaio
import multiprocessing
import ms4alg_snippets
import os
import numpy as np

processor_name = 'ms4alg_snippets.sort'
processor_version = '0.1.0'


def sort(*,
         snippet_path, geom='',
         firings_out,
         adjacency_radius, detect_sign, clip_size,
         detect_interval=10, detect_threshold=3,
         num_workers=multiprocessing.cpu_count()):
    """
    MountainSort spike sorting (version 4) - Snippets Version
    Parameters
    ----------
    snippet_path : INPUT
        M+1xN raw array containing snippets (M = #channels, N = #timepoints), the rows are the concatenated clips (across channels). The first row consist of the time (in sample number) that the snippet values were sampled at.
    geom : INPUT
        Optional geometry file (.csv format)

    firings_out : OUTPUT
        Firings array channels/times/labels/clip_index (4xL, L = num. events)

    adjacency_radius : float
        Radius of local sorting neighborhood, corresponding to the geometry file (same units). 0 means each channel is sorted independently. -1 means all channels are included in every neighborhood.
    detect_sign : int
        Use 1, -1, or 0 to detect positive peaks, negative peaks, or both, respectively
    clip_size : int
        Size of extracted clips or snippets, used throughout
    detect_threshold : float
        Threshold for event detection, corresponding to the input file. So if the input file is normalized to have noise standard deviation 1 (e.g., whitened), then this is in units of std. deviations away from the mean.
    detect_interval : int
        The minimum number of timepoints between adjacent spikes detected in the same channel neighborhood.
    num_workers : int
        Number of simultaneous workers (or processes). The default is multiprocessing.cpu_count().
    """

    tempdir = os.environ.get('ML_PROCESSOR_TEMPDIR')
    if not tempdir:
        print('Warning: environment variable ML_PROCESSOR_TEMPDIR not set. Using current directory.')
        tempdir = '.'
    print('Using tempdir={}'.format(tempdir))

    os.environ['OMP_NUM_THREADS'] = '1'

    # Read the header of the timeseries input to get the num. channels and num. timepoints
    X = mdaio.DiskReadMda(snippet_path)
    M = X.N1()-1  # Number of channels
    N = X.N2()  # Number of timepoints

    # Read the geometry file
    if geom:
        Geom = np.genfromtxt(geom, delimiter=',')
    else:
        Geom = np.zeros((M, 2))

    if Geom.shape[0] != M:
        raise Exception('Incompatible dimensions between geom and timeseries: {} != {}'.format(Geom.shape[1], M))

    MS4 = ms4alg_snippets.MountainSort4_snippets()
    MS4.setGeom(Geom)
    MS4.setSortingOpts(clip_size=clip_size, adjacency_radius=adjacency_radius, detect_sign=detect_sign,
                       detect_interval=detect_interval, detect_threshold=detect_threshold)
    MS4.setNumWorkers(num_workers)
    MS4.setSnippetPath(snippet_path)
    MS4.setFiringsOutPath(firings_out)
    MS4.setTemporaryDirectory(tempdir)
    MS4.sort()
    return True


sort.name = processor_name
sort.version = processor_version