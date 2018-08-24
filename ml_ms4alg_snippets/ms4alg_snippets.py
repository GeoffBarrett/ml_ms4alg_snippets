import numpy as np
import isosplit5
from mountainlab_pytools import mdaio
import sys
import os
import multiprocessing

# import h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()


def get_channel_neighborhood(m, Geom, *, adjacency_radius):
    M = Geom.shape[0]
    if adjacency_radius < 0:
        return np.arange(M)
    deltas = Geom - np.tile(Geom[m, :], (M, 1))
    distsqrs = np.sum(deltas ** 2, axis=1)
    inds = np.where(distsqrs <= adjacency_radius ** 2)[0]
    inds = np.sort(inds)
    return inds.ravel()


def subsample_array(X, max_num):
    if X.size == 0:
        return X
    if max_num >= len(X):
        return X
    inds = np.random.choice(len(X), max_num, replace=False)
    return X[inds]


def compute_principal_components(X, num_components):
    u, s, vt = np.linalg.svd(X)
    u = u[:, :num_components]
    return u


def compute_template_channel_peaks(templates, *, detect_sign):
    if detect_sign < 0:
        templates = templates * (-1)
    elif detect_sign == 0:
        templates = np.abs(templates)
    else:
        pass
    tc_peaks = np.max(templates, axis=1)
    tc_peak_times = np.argmax(templates, axis=1)
    return tc_peaks, tc_peak_times


def compute_sliding_maximum_snippet(snippets, radius=10):
    """This will be a sliding maximum.

    It will return a nSpikes x clip_size matrix (ret), for each
    sample it will look at samples half of the radius before and half after
    and to determine the local max for each sample.

    snippets: nSpikes x clip_size matrix"""
    ret = np.zeros_like(snippets)
    max_i = snippets.shape[-1]
    half_radius = int(radius / 2)
    for i in np.arange(snippets.shape[-1]):
        start = i - half_radius
        stop = i + half_radius
        if start < 0: start = 0
        if stop > max_i: stop = max_i
        ret[:, i] = np.amax(snippets[:, start:stop], axis=1)

    return ret.flatten()


def remove_zero_features(X):
    maxvals = np.max(np.abs(X), axis=1)
    features_to_use = np.where(maxvals > 0)[0]
    return X[features_to_use, :]


def cluster(features, *, npca):
    num_events_for_pca = np.minimum(features.shape[1], 1000)
    subsample_inds = np.random.choice(features.shape[1], num_events_for_pca, replace=False)
    u, s, vt = np.linalg.svd(features[:, subsample_inds])
    features2 = (u.transpose())[0:npca, :] @ features
    features2 = remove_zero_features(features2)
    labels = isosplit5.isosplit5(features2)
    return labels


def branch_cluster(features, *, branch_depth=2, npca=10):
    if features.size == 0:
        return np.array([])

    min_size_to_try_split = 20
    labels1 = cluster(features, npca=npca).ravel().astype('int64')
    if np.min(labels1) < 0:
        tmp_fname = '/tmp/isosplit5-debug-features.mda'
        mdaio.writemda32(features, tmp_fname)
        raise Exception('Unexpected error in isosplit5. Features written to {}'.format(tmp_fname))
    K = int(np.max(labels1))
    if K <= 1 or branch_depth <= 1:
        return labels1
    label_offset = 0
    labels_new = np.zeros(labels1.shape, dtype='int64')
    for k in range(1, K + 1):
        inds_k = np.where(labels1 == k)[0]
        if len(inds_k) > min_size_to_try_split:
            labels_k = branch_cluster(features[:, inds_k], branch_depth=branch_depth - 1, npca=npca)
            K_k = int(np.max(labels_k))
            labels_new[inds_k] = label_offset + labels_k
            label_offset += K_k
        else:
            labels_new[inds_k] = label_offset + 1
            label_offset += 1
    return labels_new


def write_firings_file(channels, times, labels, clip_inds, fname):
    L = len(channels)
    X = np.zeros((4, L), dtype='float64')
    X[0, :] = channels
    X[1, :] = times
    X[2, :] = labels
    X[3, :] = clip_inds
    mdaio.writemda64(X, fname)


def detect_on_neighborhood_from_snippets_model(X, channel_number, *, nbhd_channels, detect_threshold, detect_interval,
                                               detect_sign, clip_size, chunk_infos):
    """
        X - Snippet Model
        channel_number - The channel that you want to get the times from

    """
    t1 = chunk_infos[0]['t1']
    t2 = chunk_infos[0]['t2']

    # data_t = X.getChunk(t1=t1,t2=t2,channels=[0]).astype(np.int32)
    # we add 1 to the channel number since the first channel is the times
    # data = X.getChunk(t1=t1,t2=t2,channels=[channel_number+1]).astype(np.int32)
    data = X.getChunk(t1=t1, t2=t2, channels=nbhd_channels + 1).astype(np.int32)

    M = X.numChannels() - 1  # number of data channels
    channel_rel = np.where(nbhd_channels == channel_number)[0][
        0]  # The relative index of the central channel in the neighborhood

    if detect_sign < 0:
        # negative peaks
        np.multiply(data, -1)
    elif detect_sign == 0:
        # both negative and positive peaks
        data = np.abs(data)
    elif detect_sign > 0:
        # positive peaks
        pass

    # find the max values of the channel data (reshaped like nSpikes x clip_size)
    max_inds = np.argmax(data[channel_rel, :].reshape((-1, clip_size)), axis=1)
    clip_inds = np.arange(len(max_inds))  # also returning the clip indices

    # converting it so the indices matches the flattened data
    max_inds = np.arange(len(max_inds)) * clip_size + max_inds

    # find the max values to compare to threshold
    max_vals = data[channel_rel, :][max_inds]

    # return indices where the threshold has been reached
    threshold_bool = np.where(max_vals >= detect_threshold)[0]

    # the sample number that refers to the spike events on the chosen channel
    times = max_inds[threshold_bool]
    # the snippet index number corresponding to that event
    clip_inds = clip_inds[threshold_bool]

    # now we will calculate if the peak belongs to this channel's neighborhood
    data = data.reshape((M, -1, clip_size))

    # this will find the local neighborhood maximum for each point
    nearby_neighborhood_maximum0 = compute_sliding_maximum_snippet(np.amax(data, axis=0), radius=detect_interval)

    vals = data[channel_rel, :].flatten()[times]
    assign_to_this_neighborhood = (vals == nearby_neighborhood_maximum0[times])

    return times, clip_inds, assign_to_this_neighborhood


def compute_event_features_from_snippets(X, times, clip_ind, *, nbhd_channels, clip_size, max_num_clips_for_pca,
                                         num_features, chunk_infos):
    """compute_event_features_from_snippets
    X - the snippets model
    times - sample value of the channel peaks (for the central channel chosen)
    clip_ind - the
    """
    if times.size == 0:
        return np.array([])

    # N=X.numTimepoints()
    # X_neigh=X.getChunk(t1=0,t2=N,channels=nbhd_channels)
    M_neigh = len(nbhd_channels)

    # padding=clip_size*10

    # Subsample and extract clips for pca

    # times_for_pca=subsample_array(times,max_num_clips_for_pca)
    clips_inds_for_pca = subsample_array(clip_ind, max_num_clips_for_pca)

    t1 = chunk_infos[0]['t1']
    t2 = chunk_infos[0]['t2']

    # we add 1 to the channel number since the first channel is the times
    clips = X.getChunk(t1=t1, t2=t2, channels=nbhd_channels + 1).astype(np.int32)
    clips = clips.reshape((M_neigh, -1, clip_size))
    clips = np.swapaxes(clips, 1, 2)  # swapping axes
    clips_for_pca = clips[:, :, clips_inds_for_pca]

    # Compute the principal components
    # use twice as many features, because of branch method
    principal_components = compute_principal_components(
        clips_for_pca.reshape((M_neigh * clip_size, len(clips_inds_for_pca))),
        num_features * 2)  # (MT x 2F)

    # Compute the features for all the clips

    # projecting clip data onto the principal component axis
    features = principal_components.transpose() @ clips[:, :, clip_ind].reshape(
        (M_neigh * clip_size, len(times)))  # (2F x MT) @ (MT x L0) -> (2F x L0)

    return features


def get_real_times(X, times, *, time_channel, chunk_infos):
    """The times that we are dealing with are really just index values if we concatenated
    all the snippets to a nCh x clip_size*n_spikes matrix.

    We will bring back the time information as the 1st row of data in the .mda file represents
    the sample number that the chunk sample was recorded at (before it was removed from the continuos data).
    """

    t1 = chunk_infos[0]['t1']
    t2 = chunk_infos[0]['t2']

    # we add 1 to the channel number since the first channel is the times
    data_t = X.getChunk(t1=t1, t2=t2, channels=[time_channel]).astype(np.int32)

    return data_t[0, times]


def compute_templates_from_snippets_model(X, times, clip_ind, labels, *, nbhd_channels, clip_size, chunk_infos):
    # TODO: subsample smartly here

    M0 = len(nbhd_channels)

    t1 = chunk_infos[0]['t1']
    t2 = chunk_infos[0]['t2']

    # we add 1 to the channel number since the first channel is the times
    clips = X.getChunk(t1=t1, t2=t2, channels=nbhd_channels + 1).astype(np.int32)
    clips = clips.reshape((M0, -1, clip_size))
    clips = clips[:, clip_ind, :]
    clips = np.swapaxes(clips, 1, 2)  # swapping axes

    K = np.max(labels) if labels.size > 0 else 0
    template_sums = np.zeros((M0, clip_size, K), dtype='float64')
    template_counts = np.zeros(K, dtype='float64')

    for k in range(K):
        inds_k = np.where(labels == (k + 1))[0]
        if len(inds_k) > 0:
            template_counts[k] += len(inds_k)
            template_sums[:, :, k] += np.sum(clips[:, :, inds_k], axis=2).reshape((M0, clip_size))

    templates = np.zeros((M0, clip_size, K))
    for k in range(K):
        if template_counts[k]:
            templates[:, :, k] = template_sums[:, :, k] / template_counts[k]
    return templates


'''def create_chunk_infos(*,N,chunk_size):
    chunk_infos=[]
    num_chunks=int(np.ceil(N/chunk_size))
    for i in range(num_chunks):
        chunk={
            't1':i*chunk_size,
            't2':np.minimum(N,(i+1)*chunk_size)
        }
        chunk_infos.append(chunk)
    return chunk_infos'''


def create_chunk_infos(*, N):
    chunk_infos = []
    chunk_size = N  # I have changed it so all the data is on one chunk
    num_chunks = int(np.ceil(N / chunk_size))
    for i in range(num_chunks):
        chunk = {
            't1': i * chunk_size,
            't2': np.minimum(N, (i + 1) * chunk_size)
        }
        chunk_infos.append(chunk)
    return chunk_infos


class _NeighborhoodSorter:
    def __init__(self):
        self._sorting_opts = None
        self._clip_size = None
        self._snippets = None
        self._geom = None
        self._central_channel = None
        self._hdf5_path = None
        self._num_assigned_event_time_arrays = 0
        self._num_assigned_event_clip_ind_arrays = 0

    def setSortingOpts(self, opts):
        self._sorting_opts = opts

    def setSnippetsModel(self, model):
        self._snippets = model

    def setHdf5FilePath(self, path):
        self._hdf5_path = path

    def setGeom(self, geom):
        self._geom = geom

    def setCentralChannel(self, m):
        self._central_channel = m

    def getPhase1ClipInds(self):
        with h5py.File(self._hdf5_path, "r") as f:
            return np.array(f.get('phase1-clip_inds'))

    def getPhase1Times(self):
        with h5py.File(self._hdf5_path, "r") as f:
            return np.array(f.get('phase1-times'))

    def getPhase1ChannelAssignments(self):
        with h5py.File(self._hdf5_path, "r") as f:
            return np.array(f.get('phase1-channel-assignments'))

    def getPhase2ClipInds(self):
        with h5py.File(self._hdf5_path, "r") as f:
            return np.array(f.get('phase2-clip_inds'))

    def getPhase2Times(self):
        with h5py.File(self._hdf5_path, "r") as f:
            return np.array(f.get('phase2-times'))

    def getPhase2Labels(self):
        with h5py.File(self._hdf5_path, "r") as f:
            return np.array(f.get('phase2-labels'))

    def addAssignedEventTimes(self, times):
        with h5py.File(self._hdf5_path, "a") as f:
            f.create_dataset('assigned-event-times-{}'.format(self._num_assigned_event_time_arrays), data=times)
            self._num_assigned_event_time_arrays += 1

    def addAssignedEventClipIndices(self, clip_inds):
        with h5py.File(self._hdf5_path, "a") as f:
            f.create_dataset('assigned-event-clip_inds-{}'.format(self._num_assigned_event_clip_ind_arrays),
                             data=clip_inds)
            self._num_assigned_event_clip_ind_arrays += 1

    def runPhase1Sort(self):
        self.runSort(mode='phase1')

    def runPhase2Sort(self):
        self.runSort(mode='phase2')

    def runSort(self, *, mode):
        X = self._snippets
        M_global = X.numChannels() - 1
        N = X.numTimepoints()

        o = self._sorting_opts
        m_central = self._central_channel
        clip_size = o['clip_size']
        detect_interval = o['detect_interval']
        detect_sign = o['detect_sign']
        detect_threshold = o['detect_threshold']
        num_features = 10  # TODO: make this a sorting opt
        geom = self._geom
        if geom is None:
            geom = np.zeros((M_global, 2))

        # chunk_infos=create_chunk_infos(N=N,chunk_size=100000)
        chunk_infos = create_chunk_infos(N=N)

        nbhd_channels = get_channel_neighborhood(m_central, geom, adjacency_radius=o['adjacency_radius'])
        M_neigh = len(nbhd_channels)
        m_central_rel = np.where(nbhd_channels == m_central)[0][0]

        if mode == 'phase1':
            print('Detecting events on channel {} ({})...'.format(m_central + 1, mode));
            sys.stdout.flush()

            # these times are really indices of the peaks of the flattened channel data, if you wanted the sample index
            # relating to when the chunk was taken, you need to use this as an index of the time data (1st row of X)
            times, clip_ind, assign_to_this_neighborhood = detect_on_neighborhood_from_snippets_model(X,
                                                                                                      m_central,
                                                                                                      clip_size=clip_size,
                                                                                                      nbhd_channels=nbhd_channels,
                                                                                                      detect_threshold=detect_threshold,
                                                                                                      detect_interval=detect_interval,
                                                                                                      detect_sign=detect_sign,
                                                                                                      chunk_infos=chunk_infos)

        else:
            # get the times and clip_ind values from the phase1 sort
            times_list = []
            clip_ind_list = []
            with h5py.File(self._hdf5_path, "r") as f:
                for ii in range(self._num_assigned_event_time_arrays):
                    times_list.append(np.array(f.get('assigned-event-times-{}'.format(ii))))
                    clip_ind_list.append(np.array(f.get('assigned-event-clip_inds-{}'.format(ii))))
            times = np.concatenate(times_list) if times_list else np.array([])
            clip_ind = np.concatenate(clip_ind_list) if clip_ind_list else np.array([])

        print('Computing PCA features for channel {} ({})...'.format(m_central + 1, mode));
        sys.stdout.flush()
        max_num_clips_for_pca = 1000  # TODO: this should be a setting somewhere
        # Note: we use twice as many features, because of branch method (MT x F)
        features = compute_event_features_from_snippets(X, times, clip_ind, nbhd_channels=nbhd_channels,
                                                        clip_size=clip_size,
                                                        max_num_clips_for_pca=max_num_clips_for_pca,
                                                        num_features=num_features * 2, chunk_infos=chunk_infos)

        # The clustering
        print('Clustering for channel {} ({})...'.format(m_central + 1, mode));
        sys.stdout.flush()
        labels = branch_cluster(features, branch_depth=2, npca=num_features)
        K = np.max(labels) if labels.size > 0 else 0
        print('Found {} clusters for channel {} ({})...'.format(K, m_central + 1, mode));
        sys.stdout.flush()

        if mode == 'phase1':
            print('Computing templates for channel {} ({})...'.format(m_central + 1, mode));
            sys.stdout.flush()
            templates = compute_templates_from_snippets_model(X, times, clip_ind, labels, nbhd_channels=nbhd_channels,
                                                              clip_size=clip_size, chunk_infos=chunk_infos)

            print('Re-assigning events for channel {} ({})...'.format(m_central + 1, mode));
            sys.stdout.flush()
            # tc_peaks = the peak values for each channel in the tempaltes, tc_peak_times = index where the peaks occur
            tc_peaks, tc_peak_times = compute_template_channel_peaks(templates, detect_sign=detect_sign)  # M_neigh x K

            peak_channels = np.argmax(tc_peaks, axis=0)  # The channels on which the peaks occur

            # make channel assignments and offset times
            inds2 = np.where(assign_to_this_neighborhood)[0]
            times2 = times[inds2]
            clip_ind2 = clip_ind[inds2]
            labels2 = labels[inds2]
            channel_assignments2 = np.zeros(len(times2))
            for k in range(K):
                assigned_channel_within_neighborhood = peak_channels[k]
                dt = tc_peak_times[assigned_channel_within_neighborhood][k] - tc_peak_times[m_central_rel][k]
                inds_k = np.where(labels2 == (k + 1))[0]
                if len(inds_k) > 0:
                    times2[inds_k] += dt
                    channel_assignments2[inds_k] = nbhd_channels[assigned_channel_within_neighborhood]
                    if m_central != nbhd_channels[assigned_channel_within_neighborhood]:
                        print('Re-assigning {} events from {} to {} with dt={} (k={})'.format(len(inds_k),
                                                                                              m_central + 1,
                                                                                              nbhd_channels[
                                                                                                  assigned_channel_within_neighborhood] + 1,
                                                                                              dt, k + 1));
                        sys.stdout.flush()
            # add the phase 1 values to the hdf5 file
            with h5py.File(self._hdf5_path, "a") as f:
                f.create_dataset('phase1-times', data=times2)
                f.create_dataset('phase1-clip_inds', data=clip_ind2)
                f.create_dataset('phase1-channel-assignments', data=channel_assignments2)
        elif mode == 'phase2':
            with h5py.File(self._hdf5_path, "a") as f:
                f.create_dataset('phase2-times', data=times)
                f.create_dataset('phase2-clip_inds', data=clip_ind)
                f.create_dataset('phase2-labels', data=labels)


class SnippetModel_Hdf5:
    def __init__(self, path):
        self._hdf5_path = path
        with h5py.File(self._hdf5_path, "r") as f:
            self._num_chunks = np.array(f.get('num_chunks'))[0]
            self._chunk_size = np.array(f.get('chunk_size'))[0]
            self._padding = np.array(f.get('padding'))[0]
            self._num_channels = np.array(f.get('num_channels'))[0]
            self._num_timepoints = np.array(f.get('num_timepoints'))[0]

    def numChannels(self):
        return self._num_channels

    def numTimepoints(self):
        return self._num_timepoints

    def getChunk(self, *, t1, t2, channels):
        if (t1 < 0) or (t2 > self.numTimepoints()):
            ret = np.zeros((len(channels), t2 - t1))
            t1a = np.maximum(t1, 0)
            t2a = np.minimum(t2, self.numTimepoints())
            ret[:, t1a - (t1):t2a - (t1)] = self.getChunk(t1=t1a, t2=t2a, channels=channels)
            return ret
        else:
            c1 = int(t1 / self._chunk_size)
            c2 = int((t2 - 1) / self._chunk_size)
            ret = np.zeros((len(channels), t2 - t1))
            with h5py.File(self._hdf5_path, "r") as f:
                for cc in range(c1, c2 + 1):
                    if cc == c1:
                        t1a = t1
                    else:
                        t1a = self._chunk_size * cc
                    if cc == c2:
                        t2a = t2
                    else:
                        t2a = self._chunk_size * (cc + 1)
                    for ii in range(len(channels)):
                        m = channels[ii]
                        assert (cc >= 0)
                        assert (cc < self._num_chunks)
                        str = 'part-{}-{}'.format(m, cc)
                        offset = self._chunk_size * cc - self._padding
                        ret[ii, t1a - t1:t2a - t1] = f[str][t1a - offset:t2a - offset]
            return ret


def prepare_snippet_hdf5(snippet_fname, timeseries_hdf5_fname):
    with h5py.File(timeseries_hdf5_fname, "w") as f:
        X = mdaio.DiskReadMda(snippet_fname)
        M = X.N1()  # Number of channels
        N = X.N2()  # Number of timepoints
        chunk_size = N
        padding = 0
        chunk_size_with_padding = chunk_size + 2 * padding
        num_chunks = int(np.ceil(N / chunk_size))
        f.create_dataset('chunk_size', data=[chunk_size])
        f.create_dataset('num_chunks', data=[num_chunks])
        f.create_dataset('padding', data=[padding])
        f.create_dataset('num_channels', data=[M])
        f.create_dataset('num_timepoints', data=[N])
        for j in range(num_chunks):
            padded_chunk = np.zeros((X.N1(), chunk_size_with_padding), dtype=X.dt())
            t1 = int(j * chunk_size)  # first timepoint of the chunk
            t2 = int(np.minimum(X.N2(), (t1 + chunk_size)))  # last timepoint of chunk (+1)
            s1 = int(np.maximum(0, t1 - padding))  # first timepoint including the padding
            s2 = int(np.minimum(X.N2(), t2 + padding))  # last timepoint (+1) including the padding

            # determine aa so that t1-s1+aa = padding
            # so, aa = padding-(t1-s1)
            aa = padding - (t1 - s1)
            padded_chunk[:, aa:aa + s2 - s1] = X.readChunk(i1=0, N1=X.N1(), i2=s1, N2=s2 - s1)  # Read the padded chunk

            for m in range(M):
                f.create_dataset('part-{}-{}'.format(m, j), data=padded_chunk[m, :].ravel())


def run_phase1_sort(neighborhood_sorter):
    neighborhood_sorter.runPhase1Sort()


def run_phase2_sort(neighborhood_sorter):
    neighborhood_sorter.runPhase2Sort()


class MountainSort4_snippets:
    def __init__(self):
        self._sorting_opts = {
            "adjacency_radius": -1,
            "detect_sign": None,  # must be set explicitly
            "detect_interval": 10,
            "detect_threshold": 3,
        }
        self._snippets = None
        self._firings_out_path = None
        self._geom = None
        self._temporary_directory = None
        self._num_workers = 0

    def setSortingOpts(self, adjacency_radius=None, detect_sign=None, detect_interval=None, detect_threshold=None,
                       clip_size=None):
        if clip_size is not None:
            self._sorting_opts['clip_size'] = clip_size
        if adjacency_radius is not None:
            self._sorting_opts['adjacency_radius'] = adjacency_radius
        if detect_sign is not None:
            self._sorting_opts['detect_sign'] = detect_sign
        if detect_interval is not None:
            self._sorting_opts['detect_interval'] = detect_interval
        if detect_threshold is not None:
            self._sorting_opts['detect_threshold'] = detect_threshold

    def setSnippetPath(self, snippets_path):
        self._snippets_path = snippets_path

    def setFiringsOutPath(self, path):
        self._firings_out_path = path

    def setNumWorkers(self, num_workers):
        self._num_workers = num_workers

    def setGeom(self, geom):
        self._geom = geom

    def setTemporaryDirectory(self, tempdir):
        self._temporary_directory = tempdir

    def sort(self):
        if not self._temporary_directory:
            raise Exception('Temporary directory not set.')

        num_workers = self._num_workers
        if num_workers <= 0:
            num_workers = multiprocessing.cpu_count()

        clip_size = self._sorting_opts['clip_size']

        temp_hdf5_path = self._temporary_directory + '/snippets.hdf5'
        if os.path.exists(temp_hdf5_path):
            os.remove(temp_hdf5_path)
        '''hdf5_chunk_size=1000000
        hdf5_padding=clip_size*10
        print ('Preparing {}...'.format(temp_hdf5_path))
        prepare_timeseries_hdf5(self._timeseries_path,temp_hdf5_path,chunk_size=hdf5_chunk_size,padding=hdf5_padding)
        X=TimeseriesModel_Hdf5(temp_hdf5_path)'''
        # hdf5_chunk_size=1000000
        # hdf5_padding = 0
        # prepare_snippet_hdf5(self._snippets_path, temp_hdf5_path,chunk_size=hdf5_chunk_size,padding=hdf5_padding)
        prepare_snippet_hdf5(self._snippets_path, temp_hdf5_path)

        X = SnippetModel_Hdf5(temp_hdf5_path)

        M = X.numChannels() - 1  # the top row of data are the sample numbers
        N = X.numTimepoints()

        print('Preparing neighborhood sorters...');
        sys.stdout.flush()
        neighborhood_sorters = []

        # return self._sorting_opts, self._geom

        for m in range(M):
            NS = _NeighborhoodSorter()
            NS.setSortingOpts(self._sorting_opts)
            NS.setSnippetsModel(X)
            NS.setGeom(self._geom)
            NS.setCentralChannel(m)
            fname0 = self._temporary_directory + '/neighborhood-{}.hdf5'.format(m)
            if os.path.exists(fname0):
                os.remove(fname0)
            NS.setHdf5FilePath(fname0)
            neighborhood_sorters.append(NS)

        pool = multiprocessing.Pool(num_workers)
        pool.map(run_phase1_sort, neighborhood_sorters)

        # for each sorter it will check the assignemnts of the spikes and assign them to the respective
        # neighborhood_sorter
        for m in range(M):
            times_m = neighborhood_sorters[m].getPhase1Times()
            clip_inds_m = neighborhood_sorters[m].getPhase1ClipInds()
            channel_assignments_m = neighborhood_sorters[m].getPhase1ChannelAssignments()
            for m2 in range(M):
                inds_m_m2 = np.where(channel_assignments_m == m2)[0]
                if len(inds_m_m2) > 0:
                    neighborhood_sorters[m2].addAssignedEventTimes(times_m[inds_m_m2])
                    neighborhood_sorters[m2].addAssignedEventClipIndices(clip_inds_m[inds_m_m2])

        pool = multiprocessing.Pool(num_workers)
        pool.map(run_phase2_sort, neighborhood_sorters)

        print('Preparing output...');
        sys.stdout.flush()
        all_times_list = []
        all_labels_list = []
        all_channels_list = []
        all_clip_inds_list = []
        k_offset = 0
        for m in range(M):
            labels = neighborhood_sorters[m].getPhase2Labels()
            all_times_list.append(neighborhood_sorters[m].getPhase2Times())
            all_clip_inds_list.append(neighborhood_sorters[m].getPhase2ClipInds())
            all_labels_list.append(labels + k_offset)
            all_channels_list.append(np.ones(len(neighborhood_sorters[m].getPhase2Times())) * (m + 1))
            k_offset += np.max(labels) if labels.size > 0 else 0

        all_times = np.concatenate(all_times_list)
        all_labels = np.concatenate(all_labels_list)
        all_channels = np.concatenate(all_channels_list)
        all_clip_inds = np.concatenate(all_clip_inds_list)

        # since we are sorting by time we technically might not need
        # to do the all_clip_inds since that will sort those too
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        all_channels = all_channels[sort_inds]
        all_clip_inds = all_clip_inds[sort_inds]

        chunk_infos = create_chunk_infos(N=N)
        all_times = get_real_times(X, all_times, time_channel=0, chunk_infos=chunk_infos)

        print('Writing firings file...');
        sys.stdout.flush()
        write_firings_file(all_channels, all_times, all_labels, all_clip_inds, self._firings_out_path)

        print('Done.');
        sys.stdout.flush()