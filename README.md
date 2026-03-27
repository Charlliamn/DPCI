# DPCI: Multi-UAV Trajectory Planning and Interception

## Overview

This project implements a multi-UAV trajectory planning and interception framework in AirSim. The system integrates trajectory prediction, centerline estimation, and formation control to achieve coordinated UAV behaviors.

---

## Project Structure

```id="mka60q"
project/
├── dpci_main.py              # Core algorithm implementation
├── configs/
│   └── settings.json         # Basic AirSim configuration (environment setup only)
├── README.md
```

---

## Core Components

* **Main Algorithm**
  The core logic of the system is implemented in:

  ```
  dpci_main.py
  ```

* **Simulation Configuration**
  The file:

  ```
  configs/settings.json
  ```

  provides only a **basic AirSim environment configuration**, including vehicle setup and simulation parameters.
  It is **not part of the core algorithm logic** and can be modified depending on your AirSim environment.

---

## Model & Dataset

Due to GitHub file size limitations, the trained model and dataset are hosted on Hugging Face.

* **Model**

  ```
  unified_centerline_model.pth
  ```

* **Dataset**

  ```
  AirSimTrajectoryDataset.zip
  ```

* **Hugging Face Repository**

  ```
  https://huggingface.co/datasets/cccharlliam/SkyPath-3D
  ```

### Usage

After downloading, place the files as follows:

```id="9gz0pu"
project/
├── unified_centerline_model.pth
├── AirSimTrajectoryDataset.zip
```

---

## Installation

```id="p8lyy8"
pip install -r requirements.txt
```

---

## Running the Project

```id="gan0k9"
python dpci_main.py
```

Make sure:

1. AirSim is running
2. `settings.json` is placed correctly in the AirSim directory
3. Model file is downloaded and placed in the project root

---

## Notes

* The `settings.json` file serves as a **baseline configuration only**, and users are encouraged to adjust it according to their own simulation setup.
* This project is designed for research and experimental purposes.

---
