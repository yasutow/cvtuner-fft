import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import sys
from scipy import signal
import cv2

idx=0
CHUNK = 1024
CHUNK = 2048
CHUNK = 8192
CHUNK = 16384
RATE = 44100

name_12=np.array(["A0","A#0","B0","C1","C#1","D1","D#1","E1","F1","F#1","G1","G#1","A1","A#1","B1","C2","C#2","D2","D#2","E2","F2","F#2","G2","G#2","A2","A#2","B2","C3","C#3","D3","D#3","E3","F3","F#3","G3","G#3","A3","A#3","B3","C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4","C5","C#5","D5","D#5","E5","F5","F#5","G5","G#5","A5","A#5","B5","C6","C#6","D6","D#6","E6","F6","F#6","G6","G#6","A6","A#6","B6","C7","C#7","D7","D#7","E7","F7","F#7","G7","G#7","A7","A#7","B7","C8"])

freq_12=np.array([27.500,29.135,30.868,32.703,34.648,36.708,38.891,41.203,43.654,46.249,48.999,51.913,55.000,58.270,61.735,65.406,69.296,73.416,77.782,82.407,87.307,92.499,97.999,103.826,110.000,116.541,123.471,130.813,138.591,146.832,155.563,164.814,174.614,184.997,195.998,207.652,220.000,233.082,246.942,261.626,277.183,293.665,311.127,329.628,349.228,369.994,391.995,415.305,440.000,466.164,493.883,523.251,554.365,587.330,622.254,659.255,698.456,739.989,783.991,830.609,880.000,932.328,987.767,1046.502,1108.731,1174.659,1244.508,1318.510,1396.913,1479.978,1567.982,1661.219,1760.000,1864.655,1975.533,2093.005,2217.461,2349.318,2489.016,2637.020,2793.826,2959.955,3135.963,3322.438,3520.000,3729.310,3951.066,4186.009])

height=1000
width =2000
P = pyaudio.PyAudio()

stream  = P.open(format=pyaudio.paInt16, channels=1, rate=RATE, frames_per_buffer=CHUNK, input=True, output=True)
# capture = cv2.VideoCapture(0)

while stream.is_active():
    try:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype='int16')
        f = np.fft.fft(data, norm=None)[:int(CHUNK/2)]
        freq = np.fft.fftfreq(CHUNK,1.0e0/RATE)[:int(CHUNK/2)]
        # ret, frame = capture.read()
        background = np.ones((height, width, 3))*255
        frame=background
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        idxmax = np.argmax(np.abs(f))
        fig=plt.figure(num=None, figsize=(7,5), facecolor='white', edgecolor='black', dpi=150)
        plt.rcParams['font.size'] = 24
        ax=fig.add_subplot(1,1,1)
        ax.plot(freq, np.abs(f),color="red",lw=5)
        ax.plot(freq[idxmax],np.abs(f)[idxmax],'ro',color="red")
        ax.set_xlim([0,1000])
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer.buffer_rgba())
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        plt.close()

        name = name_12[(np.abs(freq_12 -freq[idxmax])).argmin()]
        cv2.putText(frame, name, (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(frame,"{:.1f} Hz".format(freq[idxmax]) , (100, 400), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,0), 5, cv2.LINE_AA)
        frame[height-im.shape[0]:height,width-im.shape[1]:width,:]=im
        cv2.imshow('frame',frame)
        idx=idx+1
    except KeyboardInterrupt:
        break

stream.stop_stream()
stream.close()
P.terminate()

capture.release()
cv2.destroyAllWindows()
