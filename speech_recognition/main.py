import os
import time
import torch
import json
import librosa
import random
import tqdm
import pandas as pd
import numpy as np
from op import AudioSampler, NoiseReduction, WaveToText, Decoder
from sampler import VOiCERandomSampler, VOiCEBootstrapSampler, VOiCEStratifiedSampler, Sampler
from pipeline import Evaluator, Pipeline, WordErrorRate, BatchData
import multiprocessing as mp
import pickle

DATA_SET_PATH="/data"

# knobs = [
#     ('audio_sample_rate', [8000, 10000, 12000, 14000, 16000]),
#     ('frequency_mask_width', [500, 1000, 2000, 3000, 4000]),
#     ('model', ['wav2vec2-base', 'wav2vec2-large-10m', 'wav2vec2-large-960h', 'hubert-large', 'hubert-xlarge'])
# ]

# knobs = [
#     ('audio_sample_rate', [12000, 14000, 16000]),
#     ('frequency_mask_width', [500, 2000, 4000]),
#     ('model', ['wav2vec2-base', 'wav2vec2-large-10m', 'hubert-large', 'hubert-xlarge'])
# ]

knobs = [
    ('audio_sample_rate', [14000]),
    ('frequency_mask_width', [2000]),
    ('model', ['wav2vec2-large-10m', 'hubert-large'])
]

# knobs = [
#     ('audio_sample_rate', [16000]),
#     ('frequency_mask_width', [4000]),
#     ('model', ['wav2vec2-large-960h'])
# ]

ref_df = pd.read_csv(DATA_SET_PATH + '/VOiCES_devkit/references/filename_transcripts')
ref_df.set_index('file_name', inplace=True)

def load_cache(pipe_args):
    dump_filename = '_'.join([str(i) for i in pipe_args.values()])
    with open(f"./cache/{dump_filename}.json", 'r') as f:
        dump = json.load(f)
    return dump

def get_pipeline_args():
    return {
       'audio_sample_rate': int(os.environ.get('audio_sample_rate', 8000)),
       'frequency_mask_width': int(os.environ.get('frequency_mask_width', '500')),
       'model': os.environ.get('model', 'wav2vec2-base')
    }

def random_load_speech_data():
    # randomly choose audio file for room, noise, specs
    room = random.choice(['rm1', 'rm2', 'rm3', 'rm4'])
    noise = random.choice(['babb', 'musi', 'none', 'tele'])
    path = DATA_SET_PATH + '/VOiCES_devkit/distant-16k/speech/test/' + room + '/' + noise + '/'
    sp = random.choice([f for f in os.listdir(path) if f.startswith('sp')])
    path = path + sp + '/'
    filename = random.choice(os.listdir(path))
    # Testing:
    # path = '/data/wylin2/VOiCES_devkit/distant-16k/speech/test/rm1/musi/sp5622/'
    # filename = 'Lab41-SRI-VOiCES-rm1-musi-sp5622-ch041172-sg0001-mc01-stu-clo-dg010.wav'
    # Audio
    audio, sr = librosa.load(path+filename)
    # Transcription
    transcript = ref_df.loc[filename.split('.')[0], 'transcript'] # split to remove .wav

    return audio, sr, transcript

def sample_speech_data(sampler: VOiCERandomSampler, method: str):

    filename = sampler.sample(method)  
    # /data/VOiCES_devkit/distant-16k/speech/test/rm2/none/sp1898/Lab41-SRI-VOiCES-rm2-none-sp1898-ch145702-sg0011-mc01-stu-clo-dg080.wav
    audio, sr = librosa.load(filename)
    
    # Transcription
    transcript = ref_df.loc[filename.split('/')[-1].split('.')[0], 'transcript'] # split to remove .wav

    pruned_filename = filename.split('/')[-1]
    return audio, sr, transcript, pruned_filename

def profile_pipeline(method:str):
    sampler = VOiCERandomSampler()
    pipe_args = get_pipeline_args()
    
    profile_result = {}
    # torch.cuda.reset_peak_memory_stats() 
    audio_sampler = AudioSampler(pipe_args)
    noise_reduction = NoiseReduction(pipe_args)
    wave_to_text = WaveToText(pipe_args)
    
    decoder = Decoder(pipe_args)


    print(f"Start {method} profile args: {pipe_args}")
    start_time = time.time()

    print('profile latency & input size')
    # Profile args:
    num_profile_sample_latency  = 10 #Can't be small because of warm-up
    
    for _ in tqdm.tqdm(range(num_profile_sample_latency)):
        audio, sr, transcript = random_load_speech_data()
        batch_data = {
            'audio': torch.tensor(np.expand_dims(audio, axis=0)),
            'sr': sr,
            'transcript': transcript
        }

        batch_data = audio_sampler.profile(batch_data, profile_compute_latency=True, profile_input_size=True)
        batch_data = noise_reduction.profile(batch_data, profile_compute_latency=True, profile_input_size=True)
        batch_data = wave_to_text.profile(batch_data, profile_compute_latency=True, profile_input_size=True)
        # Reset peak memory stats

        batch_data = decoder.profile(batch_data, profile_compute_latency=True, profile_input_size=True)
    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()
    # print(f"Maximum memory allocated: {max_memory_allocated / 1024 ** 2} MB")
    # print(f"Maximum memory reserved: {max_memory_reserved / 1024 ** 2} MB")
    # exit(0)
    profile_result['audio_sampler_input_size'] = audio_sampler.get_input_size()
    profile_result['noise_reduction_input_size'] = noise_reduction.get_input_size()
    profile_result['wave_to_text_input_size'] = wave_to_text.get_input_size()
    profile_result['decoder_input_size'] = decoder.get_input_size()
    profile_result['audio_sampler_compute_latency'] = audio_sampler.get_compute_latency()
    profile_result['noise_reduction_compute_latency'] = noise_reduction.get_compute_latency()
    profile_result['wave_to_text_compute_latency'] = wave_to_text.get_compute_latency()
    profile_result['decoder_compute_latency'] = decoder.get_compute_latency()
    

    print('profile accuracy')
    num_profile_sample = 10 
    batch_size = 1 # calculate cummulated accuracy every batch

    group_accuracy = [[] for i in range(16)]
    cum_accuracy = []
    # for i in tqdm.tqdm(range(num_profile_sample)):
    for i in range(num_profile_sample):
        audio, sr, transcript, filename = sample_speech_data(sampler, method)
        # audio, sr, transcript = random_load_speech_data()
        batch_data = {
            'audio': torch.tensor(np.expand_dims(audio, axis=0)),
            'sr': sr, 
            'transcript': transcript
        }
    
        batch_data = audio_sampler.profile(batch_data)
        batch_data = noise_reduction.profile(batch_data)
        batch_data = wave_to_text.profile(batch_data)
        batch_data = decoder.profile(batch_data)
        if i % batch_size == 0:
            cum_accuracy.append(decoder.get_endpoint_accuracy())
        # batch size in decoder is also 1
        sampler.feedback((filename, decoder.get_last_accuracy()))
        group_accuracy[i % 16].append(decoder.get_last_accuracy())

    profile_result['accuracy'] = decoder.get_endpoint_accuracy()
    profile_result['cummulative_accuracy'] = cum_accuracy 
    profile_result['total_profile_time'] = time.time() - start_time

    return profile_result

def bootstrap_profile_pipeline():
    pipe_args = get_pipeline_args()
    
    profile_result = {}

    audio_sampler = AudioSampler(pipe_args)
    noise_reduction = NoiseReduction(pipe_args)
    wave_to_text = WaveToText(pipe_args)
    decoder = Decoder(pipe_args)

    print(f"Profile args: {pipe_args}")
    print("start bootstraping")
    boostrap_sample_per_stratum = 10
    n_stratum = 16
    sampler = VOiCEBootstrapSampler(n_stratum * boostrap_sample_per_stratum)
    

    for i in range(n_stratum * boostrap_sample_per_stratum):
        audio, sr, transcript, filename = sample_speech_data(sampler, "feedback")
        batch_data = {
            'audio': torch.tensor(np.expand_dims(audio, axis=0)),
            'sr': sr, 
            'transcript': transcript
        }
        batch_data = audio_sampler.profile(batch_data)
        batch_data = noise_reduction.profile(batch_data)
        batch_data = wave_to_text.profile(batch_data)
        batch_data = decoder.profile(batch_data)
        sampler.feedback((filename, decoder.get_last_accuracy()))
        
    sampler.update_weight()
    
    start_time = time.time()
    
    print('profile accuracy')
    num_profile_sample = 1000
    batch_size = 1 # calculate cummulated accuracy every batch

    group_accuracy = [[] for i in range(16)]
    cum_accuracy = []
    # for i in tqdm.tqdm(range(num_profile_sample)):
    for i in range(num_profile_sample):
        audio, sr, transcript, filename = sample_speech_data(sampler, "weighted")
        # audio, sr, transcript = random_load_speech_data()
        batch_data = {
            'audio': torch.tensor(np.expand_dims(audio, axis=0)),
            'sr': sr, 
            'transcript': transcript
        }
    
        batch_data = audio_sampler.profile(batch_data)
        batch_data = noise_reduction.profile(batch_data)
        batch_data = wave_to_text.profile(batch_data)
        batch_data = decoder.profile(batch_data)
        if i % batch_size == 0:
            cum_accuracy.append(decoder.get_endpoint_accuracy())
        # batch size in decoder is also 1
        sampler.feedback((filename, decoder.get_last_accuracy()))
        group_accuracy[i % 16].append(decoder.get_last_accuracy())
    
    profile_result["group_acc"] = [np.mean(group_accuracy[i]) for i in range(16)]
    profile_result["group_std"] = [np.std(group_accuracy[i]) for i in range(16)]
    profile_result['accuracy'] = decoder.get_endpoint_accuracy()
    profile_result['cummulative_accuracy'] = cum_accuracy 
    profile_result['total_profile_time'] = time.time() - start_time

    return profile_result
   

def prepare_sampler(method: str):
    if method == "stratified_natural":
        with open("./cache/cluster_natural.json", "r") as f:
            cluster = json.load(f)
        sampler = VOiCEStratifiedSampler(cluster)
    elif method == "stratified_kmeans":
        with open("./cache/cluster_kmeans.json", "r") as f:
            cluster = json.load(f)
        sampler = VOiCEStratifiedSampler(cluster)
    else:
        with open("./cache/cluster_natural.json", "r") as f:
            cluster = json.load(f)
        keys = list(cluster.keys())
        sampler = VOiCERandomSampler(keys)
    return sampler
 
def profile_pipeline_cached(method: str, sampler: Sampler):
    method = method.split('_')[0]
    assert(method in ["bootstrap", "stratified", "random"])
    start_time = time.time()

    pipe_args = get_pipeline_args()
    print(f"Profile args: {pipe_args}")

    cache = load_cache(pipe_args)
    profile_result = {}
    audio_sampler = AudioSampler(pipe_args)
    noise_reduction = NoiseReduction(pipe_args)
    wave_to_text = WaveToText(pipe_args)
    decoder = Decoder(pipe_args)
    ev = WordErrorRate()
    
    pipeline = Pipeline(name="speech_recognition", ops=[audio_sampler, noise_reduction, wave_to_text, decoder], evaluator=ev, cache=cache)
    
    start_time = time.time()
    bt_per_stratum = 50
    n_stratum = 16
    group_accuracy = [[] for i in range(16)]
    accuracy = []
    
    print("Init done, taking time: ", time.time() - start_time)
    
    total = 6400
    if method == "bootstrap":
        nsample = total - n_stratum * bt_per_stratum

        for i in range(n_stratum * bt_per_stratum):
            filename = sampler.sample("stratified").split('/')[-1]
            result = pipeline.run_cached(filename)
            
            accuracy.append(result)
            group_accuracy[i % 16].append(result)
            
            sampler.feedback((filename, result))
        
        sampler.update_weight()
    else:
        nsample = total
    
    print("Bootstrap done, taking time: ", time.time() - start_time)
    
    def get_group_id(name) -> int:
        l1 = name.split('-')[3]
        l2 = name.split('-')[4]
        l1s = {"rm1":0, "rm2":1, "rm3":2, "rm4":3}
        l2s = {'babb':0, 'musi':1, 'none':2, 'tele':3}
        return l1s[l1] * 4 + l2s[l2]
    
    start_time = time.time()
    for i in range(nsample):
        if method == "bootstrap":
            filename = sampler.sample("weighted").split('/')[-1]
        else:
            filename = sampler.sample(method).split('/')[-1]
        result = pipeline.run_cached(filename)
        
        gid = get_group_id(filename)
        
        group_accuracy[gid].append(result)
        accuracy.append(result)

    print("Profile done, taking time: ", time.time() - start_time)
    
    cul_accuracy = [sum(accuracy[:i+1])/(i+1) for i in range(len(accuracy))]    
        
    profile_result["group_acc"] = [np.mean(group_accuracy[i]) for i in range(16)]
    profile_result["group_std"] = [np.std(group_accuracy[i]) for i in range(16)]
    profile_result['accuracy'] = accuracy
    profile_result['cummulative_accuracy'] = cul_accuracy
    # profile_result['corrected_acc'] = np.sum([np.sum(group_accuracy[i]) / total for i in range(16)])
    
    if method == "bootstrap":
        corrected_acc = 0
        len_group = total / 16
        print("Len group: ", len_group)
        for i in range(16):
            print(f"Group {i}: {np.mean(group_accuracy[i]):4f} {len(group_accuracy[i])}")
            corrected_acc += np.mean(group_accuracy[i]) * len_group / len(group_accuracy[i])
        corrected_acc /= 16
        profile_result['corrected_acc'] = corrected_acc
        print(f"Corrected acc: {profile_result['corrected_acc']}")
        
    return profile_result
      
# use Pipeline to get cache
def profile_pipeline_normal(sample_method: str):
    pipe_args = get_pipeline_args()
    dump_filename = '_'.join([str(i) for i in pipe_args.values()])

    eval_result = {}

    audio_sampler = AudioSampler(pipe_args)
    noise_reduction = NoiseReduction(pipe_args)
    wave_to_text = WaveToText(pipe_args)
    decoder = Decoder(pipe_args)
    ev = WordErrorRate()
    pipeline = Pipeline(name="speech_recognition", ops=[audio_sampler, noise_reduction, wave_to_text, decoder], evaluator=ev, cache=None)
    print(f"Bootrap profile args: {pipe_args}")
    sampler = VOiCERandomSampler()
    
    nsample = 6400
    for _ in range(nsample):
        audio, sr, transcript, filename = sample_speech_data(sampler, "random")
        batch_data = BatchData(name=filename, data={
            'audio': torch.tensor(np.expand_dims(audio, axis=0)),
            'sr': sr, 
            'transcript': transcript,
            'prediction': None,
        },label=transcript)
        result, data = pipeline.run(batch_data)
        # eval_result.update({filename: {"metric": result}})
        embedding = data["emission"]
        eval_result.update({filename: embedding})     
        print(f"Done {filename} {embedding.shape}")   
    # dump = {"args": pipe_args, "eval": "wer", "results": eval_result}    
    # with open(f"./cache/{dump_filename}.json", 'w') as f:
    #     f.write(json.dumps(dump))
    # print(f"Done, saved to {dump_filename}")
    dump = {"args": pipe_args, "results": eval_result}
    with open("./cache/embedding.pkl", 'wb') as f:
        pickle.dump(dump, f, pickle.HIGHEST_PROTOCOL)
    return 
      
# use this to get cache
def start_prepare():
    for audio_sr in knobs[0][1]:
        for freq_mask in knobs[1][1]:
            for model in knobs[2][1]:
                os.environ["audio_sample_rate"] = str(audio_sr)
                os.environ["frequency_mask_width"] = str(freq_mask)
                os.environ["model"] = str(model)
                print(f"Prepare {audio_sr} {freq_mask} {model}")
                profile_pipeline_normal("random")
                torch.cuda.empty_cache()

def start_exp(result_fname, method, num):
    if method.split("_")[0] not in ["bootstrap", "stratified", "random"]:
        raise ValueError("Method must be one of 'bootstrap', 'stratified', 'random'")
    records = []
    for audio_sr in knobs[0][1]:
        for freq_mask in knobs[1][1]:
            for model in knobs[2][1]:
                os.environ["audio_sample_rate"] = str(audio_sr)
                os.environ["frequency_mask_width"] = str(freq_mask)
                os.environ["model"] = str(model)
                for i in range(num):
                    sampler = prepare_sampler(method)
                    result = profile_pipeline_cached(method, sampler)
                    records.append(dict(
                        idx=i,
                        method=method,
                        audio_sr=audio_sr,
                        freq_mask=freq_mask,
                        model=model,
                        result=result
                    ))
                    print(f"Done {i} {audio_sr} {freq_mask} {model} {method}")
    with open(result_fname, 'w') as fp:
        records = json.dump(records, fp)  


if __name__ == "__main__":
    # num = 2
    # for method in ["random", "stratified"]:
    #     for i in range(num):
    #         start_exp(f"./result/{method}_{i}.json", method)
    
    num = 5
    date_time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    # for method in ["bootstrap", "stratified", "random"]:
    for method in ["stratified_natural", "stratified_kmeans", "random"]:
        start_exp(f"./result/{method}_{date_time_str}.json", method, num)
        print(f"Save to {method}_{date_time_str}.json")

    # start_prepare()