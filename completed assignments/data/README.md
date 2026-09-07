# Noise Lab Course Datasets

This repository contains datasets used across various Noise Lab courses. The data is organized by assignment/course topic.

## Contents

### Anomaly Detection (`anomaly-detection/`)

All files for the anomaly detection assignment:

- **demo.pcap** (2.1 MB) - Demo traffic from the University of New Brunswick's Intrusion Detection Systems dataset
- **demo.csv** (3.1 MB) - Labels for the demo traffic
- **google_home.pcap** (74 MB) - Google Home device traffic with voice queries and music playing
- **google_home_labels.txt** - Activity timestamps (voice queries, music playing start/stop)
- **nestcam_live.pcap** (9.7 MB) - Nest camera traffic with livestream and motion detection
- **nestcam_live_labels.txt** - Event timestamps (livestream, motion detection)
- **tplink_switch.pcap** (52 KB) - TP-Link smart switch traffic

### Latency Geo-Clustering (`latency-geo-clustering/`)

All files for the latency geo-clustering assignment:

- **Chicago_speedtests.csv** (23.7 MB) - Ookla speed test measurements from Chicago (January 2022 - May 2023), Comcast subscribers only
- **City_boundaries.csv** (484 KB) - Chicago city boundary data for spatial analysis

### Video Conference QoE (`video-conference-qoe/`)

All files for the video conference QoE assignment:

- **teams.csv** (32 MB) - Microsoft Teams video conference traffic data

### Video QoE (`video-qoe/`)

Files for the video QoE assignment:

- **netflix.pcap** (22 MB) - Netflix video streaming traffic capture
- **netflix_session.pkl** (1.6 KB) - Preprocessed Netflix session data
- **netflix_dataset.pkl.xz** (22 MB, compressed) - Netflix-only video quality dataset
- **video_dataset.pkl.xz** (61 MB, compressed) - Multi-service video quality dataset

**Dataset Details:**

1. **netflix_dataset.pkl**: 52,279 samples with 251 features - Netflix-only data
2. **video_dataset.pkl**: 204,713 samples with 170 features - Multi-service data (Netflix, YouTube, Twitch, Amazon Prime Video)

**Decompressing the Datasets:**

The compressed `.xz` files must be decompressed before use:

```bash
# Decompress netflix_dataset (105 MB uncompressed)
xz -d video-qoe/netflix_dataset.pkl.xz

# Decompress video_dataset (265 MB uncompressed)
xz -d video-qoe/video_dataset.pkl.xz
```

## Usage

Clone this repository to access the datasets:

```bash
git clone https://github.com/noise-courses/data.git
```

Then reference the data in your assignments using the appropriate paths.

## Attribution

- Demo data (demo.pcap, demo.csv): University of New Brunswick Intrusion Detection Systems dataset
- Smart home device traces: Noise Lab device captures
