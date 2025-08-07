#Librerias de captura de audio y visualización
import pyaudio
import wave
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

#VAD or Voice Activity Detection only works at CPU level with only 1 thread.
import torch
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 5  # segundos

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils


# Inicializar figura
fig, ax = plt.subplots()
x = np.arange(0, FRAMES_PER_BUFFER)
line, = ax.plot(x, np.random.rand(FRAMES_PER_BUFFER), lw=2)
ax.set_ylim(-30000, 30000)
ax.set_xlim(0, FRAMES_PER_BUFFER)
ax.set_title("Grabando audio en tiempo real...")
ax.set_xlabel("Muestras")
ax.set_ylabel("Amplitud")
plt.ion()
plt.show(block=False)

def grabar_con_animacion(nombre_archivo):
    frames = []
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=FRAMES_PER_BUFFER,
                     input_device_index=2
                     )

    start_time = time.time()

    while time.time() - start_time < DURATION:
        data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        line.set_ydata(audio_data)
        fig.canvas.draw()
        fig.canvas.flush_events()


    # Finalizar grabación

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Guardar WAV
    wf = wave.open(nombre_archivo, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio guardado en {nombre_archivo}")

def analizar_voz(nombre_archivo):
    wav_tensor = read_audio(nombre_archivo)
    timestamps = get_speech_timestamps(wav_tensor, model, return_seconds=True)
    if timestamps:
        print("\n🗣️ Segmentos detectados con voz:")
        for seg in timestamps:
            print(f"  • Desde {seg['start']:.2f}s hasta {seg['end']:.2f}s")
    else:
        print("\n🤐 No se detectó voz en el audio.")


if __name__ == "__main__":
    try:
        while True:
            archivo = 'tmp.wav'
            grabar_con_animacion(archivo)
            analizar_voz(archivo)
    except KeyboardInterrupt:
        print("\nFinalizando...")
        pa.terminate()
        plt.ioff()
        plt.close('all')
       
        
        










