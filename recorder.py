#####################################################
# Package:
# pip install pyaudio scipy matplotlib
#
# Usage:
# python recorder.py <folder_name> <subfolder_name (default: "")>
#
#####################################################
# https://github.com/clalanliu/Python-Multiple-Microphones-Recording/blob/master/recorder.py

# https://github.com/xiongyihui/tdoa/blob/master/realtime_tdoa.py
# https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py

import sys
import pyaudio
from scipy.io.wavfile import read as scipy_wave_read
import time
import csv
import timeit
import wave
import os
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import numpy as np

p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16

# --------global para--------
RATE = 16000
# https://stackoverflow.com/questions/58613948/what-does-the-number-of-channels-mean-in-pyaudio
CHANNELS = 1
CHUNK = 4
WIDTH = 2
FORMAT = pyaudio.paInt16
duration = 1.25  # seconds
filename_counter = 0
folder_name = "Training"
buffer_size = 200
stream_list = []
# --------global para--------

if len(sys.argv) > 1:
    folder_name = sys.argv[1]
subfolder_name = "1030"
if len(sys.argv) > 2:
    subfolder_name = sys.argv[2]


def callback(in_data, frame_count, time_info, status):
    return (in_data, pyaudio.paContinue)


def makeStream(FORMAT, CHANNELS, RATE, INDEX, CHUNK):
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=INDEX,
        frames_per_buffer=CHUNK,
    )
    return stream


def runCommand(cmand):
    if cmand == "h":
        printHelp()
    elif cmand == "d":
        setDuration()
    elif cmand == "n":
        setStartNumber()
    elif cmand == "r" or cmand == "p":
        record_utterance()
    elif cmand == "plt":
        show_last_waves()
    elif cmand == "exit":
        sys.exit()
    else:
        print("Error: Cammand not found!")


def printHelp():
    print("r: Recording n seconds audio")
    print("d: Setting duration for wav file")
    print("n: Setting starting number for filename")
    print("plt: Plot the records")
    print("exit: Exit")
    print("")


def setStartNumber():
    global filename_counter
    filename_counter = input("Name file starting at? (default 0): ")
    filename_counter = int(filename_counter)


def setDuration():
    global duration
    duration = input("Duration of wave file in seconds (default 2): ")


def gcc_phat(f_names, interp=16):
    if len(f_names) != 2:
        print("Need two mics for TDOA")
        return

    rate1, sig = scipy_wave_read(f_names[0])
    rate2, refsig = scipy_wave_read(f_names[1])

    assert rate1 == rate2

    fs = rate1
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    """

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    # print(cc)

    # find max cross correlation index
    shift = np.argmax(cc) - max_shift

    # Sometimes, there is a 180-degree phase difference between the two microphones.
    # shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    print("Time delay gcc_phat manual: ", tau)

    return tau, cc


# https://stackoverflow.com/questions/54592194/fft-of-data-received-from-pyaudio-gives-wrong-frequency
def find_TDOA(f_names):

    if len(f_names) != 2:
        print("Need two mics for TDOA")
        return

    rate1, data1 = scipy_wave_read(f_names[0])
    rate2, data2 = scipy_wave_read(f_names[1])

    assert rate1 == rate2

    # nperseg=1024?
    # Eq 7 (cross-power spectral density).
    f_Z, Z_12 = signal.csd(data1, data2, rate1, scaling="spectrum")
    f_R, R_12 = signal.csd(data1, data2, rate1, scaling="density")

    R_12_normazlied = R_12 / scipy.linalg.norm(Z_12)
    # print(R_12_normazlied)

    tau_hat = np.argmax(R_12_normazlied) / rate1
    # print(len(R_12_normazlied))
    print("Time delay: ", tau_hat)
    return tau_hat


def record_utterance():
    global duration
    global RATE
    global CHUNK
    global duration
    global stream_list
    global folder_name
    global subfolder_name
    global CHANNELS
    global filename_counter
    """
    for j in range(len(stream_list)):
        stream_list[j].start_stream()
    """
    time.sleep(0.2)

    frames = [[] for _ in range(len(stream_list))]
    start = timeit.default_timer()
    for i in range(0, int(RATE / CHUNK * duration)):
        for j in range(len(stream_list)):
            # stream_list[j].start_stream()
            data = stream_list[j].read(CHUNK, exception_on_overflow=False)
            # stream_list[j].stop_stream()
            frames[j].append(data)

    stop = timeit.default_timer()
    print("Finished: ", stop - start)

    # print(type(frames[0]))
    # print(frames[0])

    files = []

    filename = str(filename_counter) + ".wav"
    for i in range(len(stream_list)):
        f_name = os.path.join(folder_name, subfolder_name, str(i), filename)
        wf = wave.open(f_name, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames[i]))
        wf.close()

        files.append(f_name)
    """
    for j in range(len(stream_list)):
        stream_list[j].stop_stream()
    """
    filename_counter += 1

    find_TDOA(files[:2])
    gcc_phat(files[:2])


def show_last_waves():
    global folder_name
    global subfolder_name
    if filename_counter == 0:
        print("Record Not Found!")
        return
    filename = str(filename_counter - 1) + ".wav"
    for i in range(len(stream_list)):
        _fs, data = scipy_wave_read(
            os.path.join(folder_name, subfolder_name, str(i), filename)
        )
        plt.subplot(int(len(stream_list) * 100 + 10 + i + 1))
        plt.plot(data)

    plt.show()


def getDeviceInfo():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    for i in range(0, numdevices):
        if (
            p.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels")
        ) > 0:
            n = p.get_device_info_by_host_api_device_index(0, i).get("name")
            print(
                "Input Device id ", i, "-", n.encode("utf8").decode("cp950", "ignore")
            )


if __name__ == "__main__":
    getDeviceInfo()
    number_of_mics = int(input("Number of Mics: "))
    index_of_mics = []
    for i in range(number_of_mics):
        stri = "Index for " + str(i) + "-th mics."
        index_of_mics.append(int(input(stri)))
        stream_list.append(makeStream(FORMAT, CHANNELS, RATE, index_of_mics[-1], CHUNK))

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(os.path.join(folder_name, subfolder_name)):
        os.makedirs(os.path.join(folder_name, subfolder_name))
    for i in range(number_of_mics):
        if not os.path.exists(os.path.join(folder_name, subfolder_name, str(i))):
            os.makedirs(os.path.join(folder_name, subfolder_name, str(i)))

    while True:
        try:
            task_command = input("Feed me a task (h for help): ")
            runCommand(task_command)
        except KeyboardInterrupt:
            for i in range(number_of_mics):
                stream_list[i].close()
            sys.exit()
