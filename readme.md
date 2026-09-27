# Step by step guide

- What happens, when you run ```python simulate.py```, it generates ```latency_history.csv```, which contains simulated latency records as below
  ```
    timestamp,network,device_count,baseline_latency_ms,observed_latency_ms
    2025-01-01 00:00:00,3G,7,102.86,107.99
    2025-01-01 00:00:00,4G,14,46.05,63.05
    2025-01-01 00:00:00,5G,19,9.47,64.99
    2025-01-01 00:10:00,3G,37,108.57,149.63
    2025-01-01 00:10:00,4G,24,42.23,76.48
    2025-01-01 00:10:00,5G,5,7.76,16.33
    2025-01-01 00:20:00,3G,48,97.72,154.57
  ```

- Run -> ```python preprocess.py```, this will process date and prepare them for model training.
  - It pivot the simulated records (say PROCESSED_RECORD) to below format.
  ```
  Row Number	Timestamp	base_3G	base_4G	base_5G
  0 2025-01-01 00:00:00	102.86	46.05	9.47
  1 2025-01-01 00:10:00	108.57	42.23	7.76
  2 2025-01-01 00:20:00	97.72	40.24	6.59
  3 2025-01-01 00:30:00	101.31	39.43	9.43
  4 2025-01-01 00:40:00	96.11	42.40	5.98
  5 2025-01-01 00:50:00	99.10	36.91	10.86
  6 2025-01-01 01:00:00	105.55	37.89	6.40
  7 2025-01-01 01:10:00	89.59	34.71	9.77
  8 2025-01-01 01:20:00	102.08	41.94	6.80
  9 2025-01-01 01:30:00	100.50	38.62	9.35
  10 2025-01-01 01:40:00	95.18	35.12	5.80
  11 2025-01-01 01:50:00	89.83	36.76	6.87
  12 2025-01-01 02:00:00	98.78	38.54	21.21
  13 2025-01-01 02:10:00	95.11	36.22	17.24
  14 2025-01-01 02:20:00	99.71	35.39	15.22
  ```
  - Then run throw the records to generate 2 arrays. X and Y. 
  - For example, in the first iteration, it takes first 12 records, Row 0 - 11
  - It put values to X[0], where it has 12 records, each has 7 columns. Below are 7 columns
    - base_3G: The historical baseline latency for the 3G network (in milliseconds).
    - base_4G: The historical baseline latency for the 4G network (in milliseconds).
    - base_5G: The historical baseline latency for the 5G network (in milliseconds).
    - tod_sin: The sine component of the Time of Day (captures daily cyclical patterns).
    - tod_cos: The cosine component of the Time of Day (captures daily cyclical patterns).
    - dow_sin: The sine component of the Day of Week (captures weekly cyclical patterns).
    - dow_cos: The cosine component of the Day of Week (captures weekly cyclical patterns)

  - Then sample X records to 12 records groups. 
  - np[0] -> has 12 X records -> X[0] has 7 columns
  - Calculating Y[0] (The Multi-Horizon Targets), this is to supervise learning
  - The script looks ahead sequentially using HORIZONS = [1, 2, 3] and maps future entries using tgt[i + h - 1]
  - Horizon 1 (h = 1): tgt[12 + 1 - 1] -> Row 12 (02:00:00).It copies the latencies: [98.78, 38.54, 21.21] (+10 mins away).
  - Horizon 2 (h = 2): tgt[12 + 2 - 1] -> Row 13 (02:10:00).It copies the latencies: [95.11, 36.22, 17.24] (+20 mins away).
  - Horizon 3 (h = 3): tgt[12 + 3 - 1] -> Row 14 (02:20:00).It copies the latencies: [99.71, 35.39, 15.22] (+30 mins away).
  - Finally, np.concatenate flattens these vectors into a single 1D array of 9 metrics:

    ```[98.78, 38.54, 21.21, 95.11, 36.22, 17.24, 99.71, 35.39, 15.22]```



