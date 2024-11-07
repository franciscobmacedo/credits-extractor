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
3. create and populate the `movies` directory with the desired movies

## Usage

Edit the `main` function under `main.py`. You can:

- choose to extract credits from a movie given a movie file path and the credits start time (if you already know the start time of the credits)
- detect and extract all credits from movies stored under the `movies` directory (which you should create and populate)
- detect and extract credits from a particular movie

Then, you can run:

```bash
python main.py
```

The results will be under the `credits` directory with:

- One video mp4 file for the credits
- If you required the credit detection, the credit detection image boundaries (start and end) are also created. This means it's expected that the credits start in this interval.



## To improve

In `extract_credits.py`, check the `frames_have_credits` method. This is where the criteria for "is credit or not" is applied.
Extend the criteria if you want different results.

For example, we can also count the lenght of the text and determine if it's enough to be considered as credits. 
For that, we need to edit the `frame_has_text` function:

```python
def frame_has_text(self, frame: MatLike) -> bool:
        # ... Same as before
        
        # replace `if text.strip():` with this code
        if text.strip() and len(text.strip()) > 30: # Update this line to only consider "valid text" if the length of text is bigger than 30 characters
            return True
        
        # ...
```


