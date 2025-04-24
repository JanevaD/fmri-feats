# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:09:38 2024

@author: danie
"""

from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.signal as sig 
import networkx as nx 
from scipy.signal import hilbert
from scipy.linalg import eigh
from scipy.signal import spectrogram
from scipy.stats import entropy
from scipy.signal import welch



tr = 3.6

class FeatureExtraction:

    def __init__(self, time_series, hem = True):
        
        self.tr = tr
        self.time_series = time_series
        self.time_series_zs = self.time_series.copy()
        self.time_series_zs = self.time_series_zs.apply(lambda x:(x - x.mean()) / x.std(), axis = 0) # z-score timeseries

        if hem == False: 
            self.names = ['_'.join(name.split('_')[2:3]) for name in self.time_series.columns]
        elif hem == True: 
            self.names = ['_'.join(name.split('_')[1:3]) for name in self.time_series.columns]
        

            
    def get_fc(self, hem = False ):
        """
        Function to calculate the functional connectivity i.e correlation between regional BOLD signals
        
        :param time_series: regional BOLD signals
        :type time_series: DataFrame
        :
        :return: Functional Connectivity
        :rtype: DataFrame

        """
        timeseries = self.time_series_zs #z_scored timeseries 
        timeseries.columns = self.names
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([timeseries.values])[0]
        
        fc = pd.DataFrame(correlation_matrix, columns=timeseries.columns, index = timeseries.columns)
        fc.columns = self.names
        fc.index = self.names
        return fc

    def get_fc_seg_integ(self, hem = False):
        """
        Function to calculate FC Segregation and Integration of Whole Brain Networks
        
        :param fc: functional connectivity
        :type fc: DataFrame
        :param networks: network labels
        :type networks: list
        :return: functional connectivity segregation and integration
        :rtype: lists

        """
        timeseries = self.time_series_zs #z_scored timeseries 
        timeseries.columns = self.names
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([timeseries.values])[0]
        
        fc = pd.DataFrame(correlation_matrix, columns= timeseries.columns, index = timeseries.columns)
        fc.columns = self.names
        fc.index = self.names
    
        seg_fcs =[]
        integ_fcs=[]
        networks = set(self.names) 
        nets = []
        
        for network in networks:       
            nets.append(network) 
            num_columns = int(len([col for col in fc.columns if col == network]))
            
            seg_fc  = fc.copy()
            seg_fc = seg_fc.loc[network,network].sum().sum()
            seg_fcs.append(seg_fc/(num_columns*num_columns))
    
            integ_fc = fc.copy()
            integ_fc.loc[network,network]=0
            
            integ_fc = integ_fc.loc[network,::].sum().sum()
            i_n = (np.array(fc.shape[1])-num_columns)
            integ_fcs.append(integ_fc/(i_n*i_n))
            
        seg_fcs = pd.Series(seg_fcs, index = nets)
        integ_fcs = pd.Series(integ_fcs, index = nets)
        
        seg_fcs = seg_fcs.groupby(level=0).mean()
        integ_fcs= integ_fcs.groupby(level=0).mean()

        seg_fcs = seg_fcs.rename(lambda x: f"{x}_seg")
        integ_fcs = integ_fcs.rename(lambda x: f"{x}_integ")
        
        return seg_fcs, integ_fcs   


    def get_mean_dfc(self,L=15, S=2, hem = False):
        """    
        :param time_series: Regional BOLD 
        :type time_series: DataFrame
        :param L: window length 
        :type L: int
        :param S: stepsize
        :type S: int 
        :return: functional connectivity stream, stream variance and fcd
        :rtype: TYPE
        """
       
        timeseries = self.time_series_zs #z_scored timeseries 
        timeseries.columns = self.names
        
        fc_stream =[]
        dfcs = []
        M = len(timeseries)
        for i in range (0, M-S, S):
            correlation_measure = ConnectivityMeasure(kind='correlation')
            dfc = correlation_measure.fit_transform([timeseries[i:i+L].values])[0]
            dfcs.append(dfc)
            dfc_t = np.tril(dfc, k=-1).flatten()
            fc_stream.append(dfc_t)    
        
        dfcs = np.array(dfcs)   
        dfcs_mean = np.mean(dfcs,axis = 0)
        dfcs_mean = pd.DataFrame(dfcs_mean, index=self.names, columns = self.names)
        dfcs_mean_segs=[]
        dfcs_mean_integs=[]
        networks = set(self.names) 
        nets = []
        
        for network in networks:       
            nets.append(network)        
            
            num_columns = int(len([col for col in self.names if col == network]))
            
            dfcs_mean_seg  = dfcs_mean.copy()
            dfcs_mean_seg = dfcs_mean_seg.loc[network,network].sum().sum()
            dfcs_mean_segs.append(dfcs_mean_seg/(num_columns*num_columns))
    
            dfcs_mean_integ = dfcs_mean.copy()
            dfcs_mean_integ.loc[network,network]=0
            
            dfcs_mean_integ = dfcs_mean_integ.loc[network,::].sum().sum()
            i_n = (np.array(dfcs_mean.shape[1])-num_columns)
            dfcs_mean_integs.append(dfcs_mean_integ /(i_n*i_n))

        
        dfcs_mean_segs = pd.Series(dfcs_mean_segs,index = list(nets))
        dfcs_mean_integs = pd.Series(dfcs_mean_integs, index = list(nets))
        
        seg_dfcs = dfcs_mean_segs.groupby(level=0).mean()
        integ_dfcs= dfcs_mean_integs.groupby(level=0).mean()

        seg_mean_dfcs = seg_dfcs.rename(lambda x: f"{x}_mean_dseg")
        integ_mean_dfcs = integ_dfcs.rename(lambda x: f"{x}_mean_dinteg")
        return seg_mean_dfcs, integ_mean_dfcs
    
    def get_var_dfc(self,L=15, S=2, hem = False):
        """    
        :param time_series: Regional BOLD 
        :type time_series: DataFrame
        :param L: window length 
        :type L: int
        :param S: stepsize
        :type S: int 
        :return: functional connectivity stream, stream variance and fcd
        :rtype: TYPE
        """
       
        timeseries = self.time_series_zs #z_scored timeseries 
        timeseries.columns = self.names
        
        fc_stream =[]
        dfcs = []
        M = len(timeseries)
        for i in range (0, M-S, S):
            correlation_measure = ConnectivityMeasure(kind='correlation')
            dfc = correlation_measure.fit_transform([timeseries[i:i+L].values])[0]
            dfcs.append(dfc)
            dfc_t = np.tril(dfc, k=-1).flatten()
            fc_stream.append(dfc_t)    
        
        dfcs = np.array(dfcs)   
        dfcs_mean = np.mean(dfcs,axis = 0)
        dfcs_mean = pd.DataFrame(dfcs_mean, index=self.names, columns = self.names)
        dfcs_mean_segs=[]
        dfcs_mean_integs=[]
        networks = set(self.names) 
        nets = []
        
        for network in networks:       
            nets.append(network)        
            
            num_columns = int(len([col for col in self.names if col == network]))
            
            dfcs_mean_seg  = dfcs_mean.copy()
            dfcs_mean_seg = dfcs_mean_seg.loc[network,network].sum().sum()
            dfcs_mean_segs.append(dfcs_mean_seg/(num_columns*num_columns))
    
            dfcs_mean_integ = dfcs_mean.copy()
            dfcs_mean_integ.loc[network,network]=0
            
            dfcs_mean_integ = dfcs_mean_integ.loc[network,::].sum().sum()
            i_n = (np.array(dfcs_mean.shape[1])-num_columns)
            dfcs_mean_integs.append(dfcs_mean_integ /(i_n*i_n))

        
        dfcs_mean_segs = pd.Series(dfcs_mean_segs,index = list(nets))
        dfcs_mean_integs = pd.Series(dfcs_mean_integs, index = list(nets))
        
        seg_dfcs = dfcs_mean_segs.groupby(level=0).var()
        integ_dfcs= dfcs_mean_integs.groupby(level=0).var()

        seg_var_dfcs = seg_dfcs.rename(lambda x: f"{x}_var_dseg")
        integ_var_dfcs = integ_dfcs.rename(lambda x: f"{x}_var_dinteg")
        return seg_var_dfcs, integ_var_dfcs  


    def get_intra_fluidity_feats(self, L=15, S=2, hem = False):
        """    
        :param time_series: Regional BOLD 
        :type time_series: DataFrame
        :param L: window length 
        :type L: int
        :param S: stepsize
        :type S: int 
        :return: functional connectivity stream, stream variance and fcd
        :rtype: TYPE
        """
        time_series = self.time_series_zs
        time_series.columns = self.names    
        networks = set(self.names)
        time_series_c = time_series.copy()
    # time_series_c.columns = names
        time_series.columns = self.names
        fc_stream =[]
        fcd_vars = []
        fcd_means = []
        nets = []
        M = len(time_series)
        
        for network in networks:
            nets.append(network)
            fc_stream =[]
        # print (network)
            for i in range (0, M-S, S):
                correlation_measure = ConnectivityMeasure(kind='correlation')
                dfc = correlation_measure.fit_transform([np.array(time_series.loc[:,network].iloc[i:i+L].values)])[0]
                dfc_t = np.tril(dfc, k=M-S).flatten()
                fc_stream.append(dfc_t)    
                
            fcd = np.corrcoef(fc_stream)
            fcd_vars.append(np.var(np.triu(fcd, k=L-S).flatten()))
            fcd_means.append(np.mean(np.triu(fcd, k=L-S).flatten()))
            
            
        fcd_vars_intra = pd.Series(fcd_vars,index = nets)
        fcd_means_intra= pd.Series(fcd_means, index = nets)

        fcd_vars_intra = fcd_vars_intra.rename(lambda x: f"{x}_fcd_vars_intra")
        fcd_means_intra = fcd_means_intra.rename(lambda x: f"{x}_cd_means_intra")
        
        return fcd_vars_intra, fcd_means_intra

    def get_inter_fluidity_feats(self,L=15, S=2, hem = False):
        """    
        :param time_series: Regional BOLD 
        :type time_series: DataFrame
        :param L: window length 
        :type L: int
        :param S: stepsize
        :type S: int 
        :return: functional connectivity stream, stream variance and fcd
        :rtype: TYPE
        """
        timeseries = self.time_series_zs
        timeseries.columns = self.names    
        networks = set(self.names)
        time_series_c = timeseries.copy()
    # time_series_c.columns = names
        timeseries.columns = self.names
        fc_stream =[]
        fcd_vars = []
        fcd_means = []
        nets = []   
        M = len(timeseries)
        
        for network in networks:
            fc_stream =[]
        # print (network)
            nets.append(network)

            network_idx = list(timeseries.columns).index(network)
            mask = np.array(timeseries.columns) != network
            for i in range (0, M-S, S):
                correlation_measure = ConnectivityMeasure(kind='correlation')
                dfc = correlation_measure.fit_transform([np.array(timeseries.iloc[i:i+L].values)])[0]
                dfc_t = dfc[network_idx, mask].flatten()
                fc_stream.append(dfc_t)    
            fcd = np.corrcoef(fc_stream)
            fcd_vars.append(np.var(np.triu(fcd, k=L-S).flatten()))
            fcd_means.append(np.mean(np.triu(fcd, k=L-S).flatten()))
            
        fcd_vars_inter = pd.Series(fcd_vars,index = nets)
        fcd_means_inter = pd.Series(fcd_means, index = nets)

        fcd_vars_inter = fcd_vars_inter.rename(lambda x: f"{x}_fcd_vars_inter")
        fcd_means_inter = fcd_means_inter.rename(lambda x: f"{x}_cd_means_inter")
        
        return fcd_vars_inter, fcd_means_inter


    def get_falff(self, hem = False):
        
        """
        :param time_series: regional bold timeseries
        :type time_series: df
        :param tr: time repetition
        :type tr: int
        :return: functional amplitude of low frequency fluctuations
        :rtype: TYPE

        """
        # if hem == False: 
        #     names = ['_'.join(name.split('_')[2:3]) for name in time_series.columns]
    
        # elif hem == True: 
        #     names = ['_'.join(name.split('_')[1:3]) for name in time_series.columns]
        # time_series.columns = names
        alffs = []
        falffs = []
        
        timeseries = self.time_series_zs
        for i in range (timeseries.shape[1]): 
            detrended = sp.signal.detrend(timeseries.iloc[:,i])
            f, Pxx = sp.signal.welch(detrended, fs=1/tr, nperseg = 64)
    
            low_freq_indices = np.where((f >= 0.01) & (f <= 0.08))
            alff = np.sqrt(np.sum(Pxx[low_freq_indices]))
            alffs.append(alff)
            falff = alff/np.sum(Pxx)
            falffs.append(falff)
        
        alff_df = pd.DataFrame([alffs], columns=timeseries.columns)
        falff_df = pd.DataFrame([falffs], columns=timeseries.columns)

        alffs_mean = alff_df.T.groupby(level=0).mean();  alffs_mean = alffs_mean.rename(lambda x: f"{x}_alffs_mean")
        alffs_var = alff_df.T.groupby(level=0).var(); alffs_var = alffs_var.rename(lambda x: f"{x}_alffs_var")
        falffs_mean = falff_df.T.groupby(level=0).mean(); falffs_mean = falffs_mean.rename(lambda x: f"{x}_falffs_mean")
        falffs_var = falff_df.T.groupby(level=0).var(); falffs_var = falffs_var.rename(lambda x: f"{x}_falffs_var")

        return alffs_mean, alffs_var, falffs_mean, falffs_var

    def bandpass_filter(time_series, lowcut, highcut, fs=1/3.6, order=6):
        data = time_series.copy()
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = sig.butter(order, [low, high], btype='band')
        return sig.filtfilt(b, a, data, axis=0)

    def get_band_power(time_series, fl, fh):
    
        b_power = []

        for column in time_series.columns:
            time_series_filtered = bandpass_filter(time_series[column], lowcut=fl, highcut=fh) 
            b_power.append(np.sum(time_series_filtered**2))

        b_power_df = pd.DataFrame([b_power], columns = time_series.columns)
            
        b_power_df = b_power_df.T.groupby(level=0).mean().T
        b_power_df = b_power_df.rename(columns=lambda x: f"{x}_[{fl}, {fh}]_power" if x in b_power_df.columns else x)
        return b_power_df


    def calculate_phase_locking(self, time_series):
        """
        Parameters
        ----------
        time_series : pandas dataframe con
            DESCRIPTION.

        Returns
        -------
        phase_locking_matrix : TYPE
            DESCRIPTION.

        """
        """
        Calculate instantaneous phase-locking values using the Hilbert transform on parcellated BOLD time series.
        """
        analytic_signal = hilbert(time_series.values, axis=0)
        phase_data = np.angle(analytic_signal)
        
        n_regions = phase_data.shape[1]
        phase_locking_matrix = np.zeros((time_series.shape[0], n_regions, n_regions))
        
        for t in range(time_series.shape[0]):
            for i in range(n_regions):
                for j in range(n_regions):
                    phase_locking_matrix[t, i, j] = np.cos(phase_data[t, i] - phase_data[t, j])
        
        return phase_locking_matrix

    def extract_leida_modes(self, phase_locking_matrix, n_clusters=5):
        """
        Extract LEiDA modes using k-means clustering on the leading eigenvector of phase-locking matrices.
        """
        n_timepoints, n_regions, _ = phase_locking_matrix.shape
        leida_vectors = np.zeros((n_timepoints, n_regions))
        
        for t in range(n_timepoints):
            eigvals, eigvecs = eigh(phase_locking_matrix[t])
            leading_eigvec = eigvecs[:, -1] 
            leida_vectors[t] = leading_eigvec
        
        scaler = StandardScaler()
        leida_vectors = scaler.fit_transform(leida_vectors)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(leida_vectors)
        
        return labels, kmeans.cluster_centers_

    def calculate_metastability_from_modes(self, labels, phase_locking_matrix):
        """
        Calculate metastability as the variance of phase-locking within each LEiDA mode.
        """
        metastability_values = []
        unique_modes = np.unique(labels)
        
        for mode in unique_modes:
            mode_indices = np.where(labels == mode)[0]
            mode_phase_locking = phase_locking_matrix[mode_indices]
            
            variances = np.var(mode_phase_locking, axis=0).mean()
            metastability_values.append(variances)
        
        return metastability_values

    def get_metastability_leida(self):

        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        # Calculate phase-locking matrix
        
        phase_locking_matrix = self.calculate_phase_locking(timeseries)
        labels, mode_centers = self.extract_leida_modes(phase_locking_matrix)
        metastability = self.calculate_metastability_from_modes(labels, phase_locking_matrix)
        
        metastability_leida = pd.DataFrame([metastability], columns = [f"ms_mode_{i}" for i in range(metastability.shape[0])])
        return metastability_leida

    def get_metastability_plv(self):
        """
        Calculate metastability as the mean variance of instantaneous phase-locking (VAR).
        
        :param time_series: regional BOLD signals
        :type time_series: DataFrame
        :return: Metastability (mean variance of instantaneous phase-locking)
        :rtype: float
        """
        timeseries = self.time_series.copy()
        
        phase_data = np.angle(hilbert(timeseries, axis=0))

        n_regions = phase_data.shape[1]
        phase_locking_values = np.zeros((timeseries.shape[0], int(n_regions * (n_regions - 1) / 2)))
        idx = 0
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                phase_locking_values[:, idx] = np.cos(phase_data[:, i] - phase_data[:, j])
                idx += 1
        
        ms_plv = np.mean(np.var(phase_locking_values, axis=0))

        ms_plv  = pd.DataFrame([ms_plv], columns = ['ms_plv'])        
        return ms_plv
    

    def get_metastability_kop(self):
        """
        Calculate metastability as the mean variance of instantaneous phase-locking (VAR).
        
        :param time_series: regional BOLD signals
        :type time_series: DataFrame
        :return: Metastability (mean variance of instantaneous phase-locking)
        :rtype: float
        """
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        
        #subject_folder = os.path.join(output_dir, f"subject_{subject_id}")
        #os.makedirs(subject_folder, exist_ok=True)

        phase_data = np.angle(hilbert(timeseries, axis=0))

        kop = np.abs(np.mean(np.exp(1j * phase_data), axis=0))

        ms_kop = (np.var(kop, axis=0))
        ms_kop = pd.DataFrame([ms_kop], columns = ['ms_kop'])
        
        return ms_kop   


    def get_rsfa(self):
        """
        Calculate RSFA (Resting-State Fluctuation Amplitude) as the standard deviation of the time series for each region.
        """
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        rsfa = timeseries.std(axis=0)
        rsfa = rsfa.groupby(rsfa.index).mean()
        rsfa = rsfa.rename(lambda x: f"{x}_rsfa")
        return rsfa

    def get_vars(self): 
        """""
        Calculate VAR timecourse variance.

        """
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        vars = timeseries.var(axis=0)
        vars = vars.groupby(vars.index).mean()
        vars = vars.rename(index=lambda x: f"{x}_vars")

        return vars


    def get_dvars(self):
        """
        Calculate DVARS (D temporal derivative of timecourses VARiance) as the root mean square of temporal derivatives.
        """
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        temporal_derivative = np.diff(timeseries, axis=0)
        dvars = np.sqrt((temporal_derivative**2).mean(axis=0))
        
        # Convert dvars to a Pandas DataFrame before using groupby
        dvars_df = pd.DataFrame(dvars, index=timeseries.columns)
        
        # Now you can use groupby
        dvars_grouped = dvars_df.groupby(dvars_df.index).mean()
        dvars_grouped = dvars_grouped.rename(index=lambda x: f"{x}_dvars")
        
        return dvars_grouped



    def get_tv(self):
        """
        Calculate Temporal Variability (TV) as the standard deviation of ALFF over time.
        """
        # Calculate instantaneous amplitude (ALFF) using the Hilbert transform
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        analytic_signal = hilbert(timeseries, axis=0)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Temporal variability is the standard deviation of this amplitude envelope over time
        tvs = amplitude_envelope.std(axis=0)
        
        tvs = pd.DataFrame(tvs, index=timeseries.columns)

        tvs= tvs.groupby(tvs.index).mean()
        tvs = tvs.rename(index=lambda x: f"{x}_tvs")

        return tvs

    def get_spectrogram_sum(self):

        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        tr = self.tr
  
        spectrogram_sum = []
        for i, region in enumerate(timeseries.columns):
            # Generate the spectrogram
            f, t, Sxx = spectrogram(timeseries.iloc[:, i], fs=1/tr, nperseg=16)
            
            spectrogram_sum.append(np.sum(Sxx))
  
        spectrogram_sum = pd.DataFrame([spectrogram_sum], columns = timeseries.columns)
        spectrogram_sum = spectrogram_sum.T.groupby(level=0).mean()
        spectrogram_sum = spectrogram_sum.rename(columns=lambda x: f"{x}_spectrogram_sum" if x in spectrogram_sum.columns else x)   
    
        return (spectrogram_sum)


    def get_entropies(self):
        """
        Compute Shannon entropy for each region's time series.
        
        :param time_series: Regional BOLD time series
        :type time_series: DataFrame
        :return: Entropy values for each region
        :rtype: DataFrame
        """
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        entropies = [entropy(np.histogram(ts, bins=30, density=True)[0]) for ts in timeseries.T.values]
        entropies = pd.DataFrame([entropies], columns=timeseries.columns)
        entropies = entropies.T.groupby(level=0).mean()
        entropies = entropies.rename(columns=lambda x: f"{x}_entropy" if x in entropies.columns else x)

        return entropies



    def get_band_powers(self, fs=1/3.6, fl=0.01, fh=0.03):
        """
        Compute spectral power ratios for different frequency bands.

        :param fs: Sampling frequency, default is 1/3.6 Hz
        :param fl: Lower bound of the frequency range
        :param fh: Upper bound of the frequency range
        :return: Spectral power ratios for each region
        """
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        band_powers = []

        for column in timeseries.columns:
            # Compute the power spectral density using Welch's method
            f, Pxx = welch(timeseries[column], fs=fs, nperseg=64)
            
            # Find indices where frequency is within the specified range
            band_indices = (f >= fl) & (f <= fh)

            # Sum the power values for the frequency range
            band_power = np.sum(Pxx[band_indices])

            band_powers.append(band_power)

        # Convert list to a DataFrame
        band_powers = pd.DataFrame([band_powers], columns=self.names)
        band_powers = band_powers.T.groupby(level=0).mean()
        band_powers = band_powers.rename(columns=lambda x: f"{x}_band_power")
        
        return band_powers


    def get_hurst_exponent(self):
        """
        Compute the Hurst exponent for each time series to assess self-similarity.
        
        :param time_series: Regional BOLD time series
        :type time_series: DataFrame
        :return: Hurst exponent values for each region
        :rtype: DataFrame
        """
        timeseries = self.time_series.copy()
        timeseries.columns = self.names
        # Hurst exponent calculation
        def hurst(ts):
            lags = np.arange(2, 20)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]
        
        hurst_values = [hurst(timeseries[col]) for col in timeseries.columns]
        hurst_values = pd.DataFrame([hurst_values], columns=timeseries.columns)
        hurst_values = hurst_values.T.groupby(level=0).mean()
        hurst_values = hurst_values.rename(columns=lambda x: f"{x}_hurst_exp" if x in hurst_values.columns else x)

        return hurst_values


    def get_graph_measures(self, threshold=0.2):
        """
        Compute graph metrics for each brain network and return as a single-row DataFrame.

        :param threshold: Minimum correlation value to consider an edge
        :return: DataFrame (1 row, columns = graph metrics for each network)
        """

        timeseries = self.time_series_zs.copy()
        timeseries.columns = self.names
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([timeseries.values])[0]

        fc = pd.DataFrame(correlation_matrix, columns=self.names, index=self.names)

        networks = set(self.names)
        
        graph_metrics = {}

        for network in networks:

            regions = [col for col in fc.columns if col == network]

            if len(regions) < 2:  
                continue

            sub_fc = fc.loc[regions, regions]

            sub_fc_thresh = sub_fc.copy()
            sub_fc_thresh[np.abs(sub_fc_thresh) < threshold] = 0

            G = nx.from_pandas_adjacency(sub_fc_thresh)

            graph_metrics[f"{network}_mean_degree"] = np.mean(list(dict(G.degree()).values()))
            graph_metrics[f"{network}_clustering_coefficient"] = nx.average_clustering(G)
            graph_metrics[f"{network}_global_efficiency"] = nx.global_efficiency(G)
            graph_metrics[f"{network}_local_efficiency"] = (
                np.mean([nx.local_efficiency(G) for _ in G.nodes]) if len(G.nodes) > 1 else 0
            )
            graph_metrics[f"{network}_modularity"] = (
                nx.algorithms.community.quality.modularity(G, list(nx.algorithms.community.greedy_modularity_communities(G)))
                if nx.number_of_nodes(G) > 1 else 0
            )
        
        graph_metrics = pd.DataFrame([graph_metrics])

        return graph_metrics