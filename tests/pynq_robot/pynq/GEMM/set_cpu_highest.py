"""Request the highest supported CPUFreq policy and report actual clocks."""
from runtime.neon_runtime import highest_cpu_environment, frequency_report

if __name__ == '__main__':
    with highest_cpu_environment(keep_highest=True) as configuration:
        print(frequency_report(configuration))
