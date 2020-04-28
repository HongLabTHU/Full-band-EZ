import numpy as np
from scipy.signal import butter,filtfilt,lfilter


def compute_hfer(target_data,base_data,fs):
    target_sq=target_data**2
    base_sq=base_data**2
    window=int(fs/2.0)
    target_energy=lfilter(np.ones(window),1,target_sq,axis=-1)
    base_energy=lfilter(np.ones(window),1,base_sq,axis=-1)
    target_energy=target_energy[:,int(fs/2-1):]
    base_energy=base_energy[:,int(fs/2-1):]
    base_energy_ref=np.sum(base_energy,axis=1)/base_energy.shape[1]
    target_de_matrix=base_energy_ref[:,np.newaxis]*np.ones((1,target_energy.shape[1]))
    base_de_matrix=base_energy_ref[:,np.newaxis]*np.ones((1,base_energy.shape[1]))
    norm_target_energy=target_energy/target_de_matrix.astype(np.float32)
    norm_base_energy=base_energy/base_de_matrix.astype(np.float32)
    return norm_target_energy,norm_base_energy

def determine_threshold_onset(target,base):
    base_data=base.copy()
    target_data=target.copy()
    sigma=np.std(base_data,axis=1,ddof=1)
    channel_max_base=np.max(base_data,axis=1)
    thresh_value=channel_max_base+20*sigma
    onset_location=np.zeros(shape=(target_data.shape[0],))
    for channel_idx in range(target_data.shape[0]):
        logic_vec=target_data[channel_idx,:]>thresh_value[channel_idx]
        if np.sum(logic_vec)==0:
            onset_location[channel_idx]=len(logic_vec)
        else:
            onset_location[channel_idx]=np.where(logic_vec!=0)[0][0]
    return onset_location

def compute_ei_index(target,base,fs):
    channel_onset=determine_threshold_onset(target,base)
    # print channel_onset
    seizure_location=np.min(channel_onset)
    onset_channel=np.argmin(channel_onset)
    hfer=np.sum(target[:,int(seizure_location):int(seizure_location+0.25*fs)],axis=1)/(fs*0.25)
    onset_asend=np.sort(channel_onset)
    time_rank_tmp=np.argsort(channel_onset)
    onset_rank=np.argsort(time_rank_tmp)+1
    onset_rank=np.ones((onset_rank.shape[0],))/np.float32(onset_rank)
    ei=np.sqrt(hfer*onset_rank)
    for i in range(len(ei)):
        if np.isnan(ei[i]) or np.isinf(ei[i]):
            ei[i]=0
    ei=ei/np.max(ei)
    return ei,hfer,onset_rank
