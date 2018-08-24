from mountainlab_pytools import mdaio
import numpy as np
import multiprocessing
import time
import os

processor_name = 'ms4alg_snippets.whiten'
processor_version = '0.1.0'


class SharedChunkInfo():
    def __init__(self, num_chunks):
        self.timer_timestamp = multiprocessing.Value('d', time.time(), lock=False)
        self.last_appended_chunk = multiprocessing.Value('l', -1, lock=False)
        self.num_chunks = num_chunks
        self.num_completed_chunks = multiprocessing.Value('l', 0, lock=False)
        self.lock = multiprocessing.Lock()

    def reportChunkCompleted(self, num):
        with self.lock:
            self.num_completed_chunks.value += 1

    def reportChunkAppended(self, num):
        with self.lock:
            self.last_appended_chunk.value = num

    def lastAppendedChunk(self):
        with self.lock:
            return self.last_appended_chunk.value

    def resetTimer(self):
        with self.lock:
            self.timer_timestamp.value = time.time()

    def elapsedTime(self):
        with self.lock:
            return time.time() - self.timer_timestamp.value

    def printStatus(self):
        with self.lock:
            print('Processed {} of {} chunks...'.format(self.num_completed_chunks.value, self.num_chunks))


def compute_AAt_matrix_for_chunk(num):
    opts = g_opts
    in_fname = opts['timeseries']  # The entire (large) input file
    # out_fname = opts['timeseries_out']  # The entire (large) output file
    chunk_size = opts['chunk_size']

    X = mdaio.DiskReadMda(in_fname)

    t1 = int(num * opts['chunk_size'])  # first timepoint of the chunk
    t2 = int(np.minimum(X.N2(), (t1 + chunk_size)))  # last timepoint of chunk (+1)

    # Ensuring that this chunk value is float64 to avoid svd complications
    # we have to make sure to remove the first row which are time values
    chunk = X.readChunk(i1=0, N1=X.N1(), i2=t1, N2=t2 - t1)[1:, :].astype(np.float32)  # Read the chunk

    ret = chunk @ np.transpose(chunk)

    return ret


def whiten_chunk(num, W):
    # print('Whitening {}'.format(num))
    opts = g_opts
    # print('Whitening chunk {} of {}'.format(num,opts['num_chunks']))
    in_fname = opts['timeseries']  # The entire (large) input file
    out_fname = opts['timeseries_out']  # The entire (large) output file
    chunk_size = opts['chunk_size']

    X = mdaio.DiskReadMda(in_fname)

    t1 = int(num * opts['chunk_size'])  # first timepoint of the chunk
    t2 = int(np.minimum(X.N2(), (t1 + chunk_size)))  # last timepoint of chunk (+1)

    chunk = X.readChunk(i1=0, N1=X.N1(), i2=t1, N2=t2 - t1)  # Read the chunk

    # ensuring that the time points aren't whitened
    chunk[1:, :] = W @ chunk[1:, :]

    ###########################################################################################
    # Now we wait until we are ready to append to the output file
    # Note that we need to append in order, thus the shared_data object
    ###########################################################################################
    g_shared_data.reportChunkCompleted(num)  # Report that we have completed this chunk
    while True:
        if num == g_shared_data.lastAppendedChunk() + 1:
            break
        time.sleep(0.005)  # so we don't saturate the CPU unnecessarily

    # Append the filtered chunk (excluding the padding) to the output file
    mdaio.appendmda(chunk, out_fname)

    # Report that we have appended so the next chunk can proceed
    g_shared_data.reportChunkAppended(num)

    # Print status if it has been long enough
    if g_shared_data.elapsedTime() > 4:
        g_shared_data.printStatus()
        g_shared_data.resetTimer()


def whiten(*, timeseries, timeseries_out,
           clip_size=50, num_clips_per_chunk=6000, num_processes=os.cpu_count()
           ):
    """
    Whiten a multi-channel timeseries

    Parameters
    ----------
    timeseries : INPUT
        MxN raw timeseries array (M = #channels, N = #timepoints)

    timeseries_out : OUTPUT
        Whitened output (MxN array)

    clip_size : int
        The number of samples for each chunk that was saved (default 50).
    num_clips_per_chunk : int
        The data will be whitened in chunks, choose the number of clips per chunk (default 6000).
    num_processes : int
        The number of processes to use when performing multiprocessing.
    """

    chunk_size = int(clip_size * num_clips_per_chunk)

    X = mdaio.DiskReadMda(timeseries)
    M = X.N1()  # Number of channels (+1 since time is the 1st row)
    N = X.N2()  # Number of timepoints

    # chunk_size=N # right now we are putting all the data into one chunk

    num_chunks_for_computing_cov_matrix = 10

    num_chunks = int(np.ceil(N / chunk_size))
    print('Chunk size: {}, Num chunks: {}, Num processes: {}'.format(chunk_size, num_chunks, num_processes))

    opts = {
        "timeseries": timeseries,
        "timeseries_out": timeseries_out,
        "chunk_size": chunk_size,
        "num_processes": num_processes,
        "num_chunks": num_chunks
    }
    global g_opts
    g_opts = opts

    pool = multiprocessing.Pool(processes=num_processes)
    step = int(np.maximum(1, np.floor(num_chunks / num_chunks_for_computing_cov_matrix)))
    AAt_matrices = pool.map(compute_AAt_matrix_for_chunk, range(0, num_chunks, step), chunksize=1)

    AAt = np.zeros((M - 1, M - 1), dtype='float64')

    for M0 in AAt_matrices:
        # important: need to fix the denominator here to account for possible smaller chunk
        AAt += M0 / (len(AAt_matrices) * chunk_size)

    U, S, Ut = np.linalg.svd(AAt, full_matrices=True)

    W = (U @ np.diag(1 / np.sqrt(S))) @ Ut
    # print ('Whitening matrix:')
    # print (W)

    global g_shared_data
    g_shared_data = SharedChunkInfo(num_chunks)
    mdaio.writemda32(np.zeros([M, 0]), timeseries_out)

    pool = multiprocessing.Pool(processes=num_processes)
    pool.starmap(whiten_chunk, [(num, W) for num in range(0, num_chunks)], chunksize=1)

    return True


whiten.name = processor_name
whiten.version = processor_version
whiten.author = "Geoffrey Barrett"
