from typing import List
import io


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from scipy.io import wavfile
import scipy
from loguru import logger



MODEL_PATH="https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1"

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    try:
        waveform = scipy.signal.resample(waveform, desired_length)
    except Exception as e:
        logger.debug(e) 
        from scipy import signal
        waveform = signal.resample(waveform, desired_length) 
  return desired_sample_rate, waveform


def handle_audio_file_to_numpy(file_path:str)->np.ndarray:
    sample_rate, wav_data = wavfile.read(file_path, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    waveform = wav_data / tf.int16.max
    # just using 1 channel for simple version
    return waveform[:,0]

def init_module():
    model = hub.load(MODEL_PATH)
    return model

def extract_feature_full(audio_file_path:str,model)->List[float]:
    
    waveform=handle_audio_file_to_numpy(audio_file_path)
    # Run the model, check the output.
    
    scores, embeddings, log_mel_spectrogram = model(waveform)
    scores.shape.assert_is_compatible_with([None, 521])
    embeddings.shape.assert_is_compatible_with([None, 1024])
    log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])
    
    return scores.numpy().tolist()
    
    