import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import fft, fftfreq
from datetime import datetime
import os

# ---- تابع محاسبه SNR (Signal-to-Noise Ratio) ----
# SNR = نسبت توان سیگنال به توان نویز (dB)
def snr_db(signal, noise):
    return 10 * np.log10(np.var(signal) / np.var(noise))

# ---- تابع اصلی پردازش سیگنال ----
def process_signal(signal, Fs, lowcut, highcut, notch_freq=50, Q=30):
    """
    پردازش سیگنال تک کاناله
    پارامترها:
        signal      : سیگنال ورودی (ECG یا EMG)
        Fs        : نرخ نمونه‌برداری (Sampling Frequency)
        lowcut      : حد پایین باند فیلتر میان‌گذر (Hz)
        highcut     : حد بالای باند فیلتر میان‌گذر (Hz)
        notch_freq  : فرکانس فیلتر Notch برای حذف نویز (معمولاً 50Hz)
        Q    : کیفیت فیلتر Notch (ضریب تشدید)
        
    عملیات انجام شده:
        1- رسم نمودار حوزه زمان سیگنال خام و فیلتر شده
        2- محاسبه و رسم طیف فرکانسی (FFT)
        3- اعمال فیلتر Bandpass و Notch
        4- محاسبه SNR و چاپ گزارش تحلیل
    """

    N = len(signal)                          # تعداد نمونه‌ها
    t = np.linspace(0, N / Fs, N)           # محور زمان

    # ---- FFT سیگنال خام ----
    fft_raw = fft(signal)                    # تبدیل فوریه
    freqs = fftfreq(N, 1 / Fs)              # محور فرکانس

    # ---- فیلتر میان‌گذر (Bandpass) ----
    # فیلتر Butterworth مرتبه 4
    # این فیلتر یک سیستم LTI است
    b, a = butter(4, [lowcut / (Fs / 2), highcut / (Fs / 2)], btype='band')
    filtered = filtfilt(b, a, signal)       # اعمال فیلتر با روش فوروارد-بک‌وارد

    # ---- فیلتر Notch برای حذف نویز برق 50Hz ----
    b_notch, a_notch = iirnotch(notch_freq, Q, Fs)
    filtered = filtfilt(b_notch, a_notch, filtered)

    # ---- FFT سیگنال فیلتر شده ----
    fft_filtered = fft(filtered)

    # ---- محاسبه نویز و SNR ----
    noise_component = signal - filtered
    snr_value = snr_db(filtered, noise_component)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # برای ذخیره فایل‌ها با زمان

    # ---- نمودار حوزه زمان ----
    plt.figure(figsize=(12, 6))

    # نمودار سیگنال خام
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label="Raw Signal")
    plt.title("Raw Signal (Time Domain)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    # نمودار سیگنال فیلتر شده
    plt.subplot(2, 1, 2)
    plt.plot(t, filtered, label="Filtered Signal", color='orange')
    plt.title(f"Filtered Signal (Time Domain) | SNR: {snr_value:.2f} dB")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"time_domain_{timestamp}.png", dpi=300)  # ذخیره تصویر
    plt.show()

    # ---- نمودار حوزه فرکانس (FFT) ----
    plt.figure(figsize=(12, 5))
    plt.plot(freqs[:N // 2], np.abs(fft_raw[:N // 2]), label="Raw Spectrum")
    plt.plot(freqs[:N // 2], np.abs(fft_filtered[:N // 2]), label="Filtered Spectrum", color='orange')
    plt.title("Frequency Domain (FFT)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()
    plt.savefig(f"frequency_domain_{timestamp}.png", dpi=300)
    plt.show()

    # ---- گزارش تحلیلی کوتاه ----
    print("\n--- Signal Analysis Report ---")
    print(f"Signal length: {N} samples")
    print(f"Sampling rate: {Fs} Hz")
    print(f"Bandpass filter range: {lowcut}-{highcut} Hz")
    print(f"Notch filter: {notch_freq} Hz, Q={Q}")
    print(f"Estimated SNR after filtering: {snr_value:.2f} dB")
    print("Note: Noise component likely shows peak at 50Hz if present.")
    print("-------------------------------\n")


# -----------------------------
# بخش اصلی اجرای برنامه
# -----------------------------
if __name__ == "__main__":

    Fs = 1000  # نرخ نمونه‌برداری (نمونه/ثانیه)
    T = 5      # مدت زمان سیگنال (ثانیه)
    t = np.linspace(0, T, Fs * T)

    # انتخاب نوع سیگنال توسط کاربر
    signal_type = input("Choose signal type (ecg / emg): ").lower()
    csv_file = input("Enter CSV file path (or press Enter to generate synthetic signal): ")

    # ===============================
    # خواندن فایل واقعی CSV اگر وجود داشت
    # ===============================
    if csv_file and os.path.exists(csv_file):
        signal = np.loadtxt(csv_file, delimiter=",")
        print("CSV file loaded successfully.")

        # تنظیم باند فرکانسی بر اساس نوع سیگنال
        if signal_type == "ecg":
            lowcut, highcut = 0.5, 40
        else:
            lowcut, highcut = 20, 450

    # ===============================
    # تولید سیگنال مصنوعی در صورت عدم وجود فایل
    # ===============================
    else:
        print("Generating synthetic signal...")

        if signal_type == "ecg":
            # ---- تولید سیگنال ECG ساده ----
            heart_rate = 72                 # ضربان قلب بر حسب bpm
            f_hr = heart_rate / 60          # فرکانس ضربان بر حسب Hz
            signal = (
                1.2 * np.sin(2 * np.pi * f_hr * t) +       # موج اصلی R
                0.25 * np.sin(2 * np.pi * 2 * f_hr * t) +  # هارمونیک دوم
                0.15 * np.sin(2 * np.pi * 3 * f_hr * t) +  # هارمونیک سوم
                0.4 * np.sin(2 * np.pi * 50 * t) +         # نویز برق 50Hz
                0.05 * np.random.randn(len(t))             # نویز سفید تصادفی
            )
            lowcut, highcut = 0.5, 40

        elif signal_type == "emg":
            # ---- تولید سیگنال EMG ساده ----
            signal = (
                0.8 * np.random.randn(len(t)) * np.sin(2 * np.pi * 80 * t) +
                0.6 * np.random.randn(len(t)) * np.sin(2 * np.pi * 120 * t) +
                0.4 * np.sin(2 * np.pi * 50 * t)
            )
            lowcut, highcut = 20, 450

        else:
            raise ValueError("Invalid signal type! Choose 'ecg' or 'emg'.")

    # ---- پردازش و تحلیل سیگنال ----
    process_signal(signal, Fs, lowcut, highcut)