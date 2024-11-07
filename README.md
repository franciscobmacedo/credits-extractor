# Movie Credits Extractor

This script extracts the credits from movies stored in a specified directory. It supports any video format and can either use a known start time for the credits or detect the start time automatically.

## Requirements

- Python 3.11+


## Setup

1. Clone the repository or download the code.
2. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
3. create and populate the "movies" directory with the desired movies

## Usage

Edit the `main` function under `main.py`. You can:

- choose to extract credits from a movie given a movie file path and the credits start time (if you already know the start time of the credits)
- detect and extract all credits from movies stored under the "movies" directory (which you should create and populate)
- - detect and extract credits from a particular movie

```bash
python main.py

```

