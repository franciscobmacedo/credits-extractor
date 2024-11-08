from typing import Union

from extract_credits import MovieCreditsExtractor, CreditsNotFoundError
import os
import pathlib

# where the movies are stored - this directory should contain the video files, ANY video format is supported
MOVIES_DIR = "movies"

# where the output credits will be stored - this directory will be created if it doesn't exist
OUTPUT_DIR = "credits"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_credits(movie_path: Union[str, pathlib.Path], start_time: str):
    """
    TIP: Use this function when you know the start time of the credits in the movie.

    Extracts the credits from a video file given the start time of the credits.
    Saves the credits to a new video file in the output directory.
    """
    if isinstance(movie_path, str):
        movie_path = pathlib.Path(movie_path)

    # create the MovieCreditsExtractor object
    extractor = MovieCreditsExtractor(
        input_path=movie_path,
        output_directory=OUTPUT_DIR,
    )

    # extract the credits
    extractor.export_credits_video(start_time=start_time)


def detect_and_export_credits(movie_path: Union[str, pathlib.Path]):
    """
    TIP: Use this function when you don't know the start time of the credits in the movie.

    Detects the start time of the credits in a movie and exports the credits to a new video file.
    """
    if isinstance(movie_path, str):
        movie_path = pathlib.Path(movie_path)

    print(f"\nExtracting credits from {movie_path}")
    extractor = MovieCreditsExtractor(
        input_path=movie_path,
        output_directory=OUTPUT_DIR,
    )
    analysis_batch_seconds = 100
    required_dark_seconds = 20
    # get the start frame of the boundary of the credits
    start_frame = extractor.get_credits_start_frame(
        frames_to_skip=1,  # number of frames to skip when detecting credits. Higher values will make the process faster but less accurate. Take into account when setting "required_dark_seconds" and "required_text_seconds" and "analysis_batch_seconds".
        minutes_before_end=10,  # number of minutes before the end of the movie to start running the credits detection. This is to avoid analyzing the whole movie.
        analysis_batch_seconds=analysis_batch_seconds,  # Interval in seconds for each batch of frames that will be analysed for credits. Lower values will give a credits start time more accurate, but also affect the granularity of required_dark_seconds and required_text_seconds.
        required_dark_seconds=required_dark_seconds,  # number of seconds (with frames_to_skip as an interval) with dark frames to consider that a batch of frames has credits in it. This depends on the movie FPS, "frames_to_skip" and their relationship.
        required_text_seconds=2,  # number of seconds with text (when there's a dark background) to consider that a batch of frames has credits in it. This depends on the movie FPS, "frames_to_skip", "required_dark_seconds"   parameter and their relationship.
        gray_treshold=10,
    )
    # export the credits video
    extractor.export_credits_video(
        start_frame=start_frame, 
        initial_buffer_seconds=analysis_batch_seconds - required_dark_seconds # this is to avoid cutting the first seconds of the credits
    )


def detect_and_export_credits_for_all_movies():
    """
    TIP: Use this function when you don't know the start time of the credits in the movies AND you want to extract credits from all movies in the movies directory.

    Detects the start time of the credits in all movies in the movies directory and exports the credits to a new video file.
    """
    for movie in pathlib.Path(MOVIES_DIR).iterdir():
        try:
            detect_and_export_credits(movie_path=movie)
        except CreditsNotFoundError as e:
            print(f"credits not found in movie {movie} (skiping this one): {e}")
            continue
        except Exception as e:
            print(
                f"Unknown error extracting credits from movie {movie} (skiping this one): {e}"
            )
            continue


def main():
    # to extract credits from a single movie, when you know the start time of the credits
    # extract_credits(
    #  os.path.join(MOVIES_DIR, "Five.Easy.Pieces.1970.PROPER.1080p.BluRay.x264-SADPANDA.mkv"),
    #     "00:58:55"
    # )

    # to detect and extract credits from a particular movies
    # detect_and_export_credits(
    #     movie_path=os.path.join(
    #         MOVIES_DIR, "Five.Easy.Pieces.1970.PROPER.1080p.BluRay.x264-SADPANDA.mkv"
    #     )
    # )

    # to detect and extract credits from all movies in the movies directory
    detect_and_export_credits_for_all_movies()


if __name__ == "__main__":
    # This is the entry point of the script
    # It runs when the script is executed from the command line like this:
    # python main.py

    main()
