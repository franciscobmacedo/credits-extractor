import copy
import pathlib
import re
import time
from typing import Generator, Optional, Union
import cv2
from cv2.typing import MatLike
import numpy as np
import pytesseract


class CreditsNotFoundError(Exception):
    pass
class MovieCreditsExtractor:
    def __init__(
        self,
        input_path: Union[str, pathlib.Path],
        output_directory: Union[str, pathlib.Path],
    ):
        """
        Initialize the MovieCreditsExtractor object with the input video path and other parameters.
        input_path (Union[str,pathlib.Path]): Path to the input MP4 video file.
        """
        self.input_path = input_path
        self.output_directory = output_directory

        self.cap = cv2.VideoCapture(input_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = self.total_frames / self.fps

        duration_display = self.get_time_from_frame(self.total_frames)

        print(
            f"""
            -----------------
            VIDEO PROPERTIES
            -----------------
            Total Duration: {duration_display} ({self.total_duration} seconds)
            Resolution: {self.width}x{self.height}
            FPS: {self.fps}
            Total Frames: {self.total_frames}
        """
        )

    def __del__(self):
        self.cap.release()

    def get_time_from_frame(self, frame_number: int) -> str:
        """
        return time in hours, minutes, seconds from frame number
        """

        time_in_seconds = frame_number / self.fps
        hours = time_in_seconds // 3600
        minutes = (time_in_seconds % 3600) // 60
        seconds = time_in_seconds % 60

        hours_str = str(int(hours)).zfill(2)
        minutes_str = str(int(minutes)).zfill(2)
        seconds_str = str(int(seconds)).zfill(2)
        return f"{hours_str}:{minutes_str}:{seconds_str}"

    def get_frame_image_from_video(self, frame_number: int) -> np.ndarray:
        """
        Get the image of a frame from the video at the specified frame number.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Error reading frame {frame_number}")
        return frame

    def frame_is_dark(self, frame: MatLike, gray_treshold: int) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        return (
            mean_intensity < gray_treshold
        )  # Adjust if necessary based on actual video brightness

    def frame_has_text(self, frame: MatLike) -> bool:
        """
        Function to detect text in a given frame.
        Replace with your actual logic for text detection.
        """
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to detect text
        try:
            text = pytesseract.image_to_string(gray)
            if text.strip():
                return True
        except Exception as e:
            print("Error in detecting text (ignoring):", e)
            pass

        return False

    def frames_have_credits(
        self,
        gray_treshold: int,
        frames: Generator[MatLike, None, None],
        frame_count_to_analyze: int,
        required_dark_seconds: Optional[int] = None,
        required_text_seconds: Optional[int] = None,
    ) -> bool:
        """
        Function to detect if a given set of frames have movie credits or not.
        Count the number of dark frames and text frames in the given set of frames.
        Check if the number of dark frames and text frames meet the required criteria.
        """
        dark_frame_count = text_frame_count = 0
        required_dark_frames = (
            required_dark_seconds * self.fps if required_dark_seconds else None
        )
        required_text_frames = (
            required_text_seconds * self.fps if required_text_seconds else None
        )

        if required_dark_frames is not None:
            # we need to analyze the frames until we find the required number of dark frames or give up if
            # the remaining frames are less than the required number of dark frames
            frame_limit_to_give_up = frame_count_to_analyze - required_dark_frames
        elif required_text_frames is not None:
            # If we are just checking text, we need to analyze the frames until we find the required number of text frames or give up if
            # the remaining frames are less than the required number of text frames
            frame_limit_to_give_up = frame_count_to_analyze - required_text_frames
        else:
            # Otherwise, we can analyze all the frames
            frame_limit_to_give_up = frame_count_to_analyze
        for frame_position, frame in enumerate(frames):
            if required_dark_frames is not None and self.frame_is_dark(
                frame, gray_treshold
            ):
                dark_frame_count += 1
                # if frame is dark, check if it has text
                if required_text_frames is not None and self.frame_has_text(frame):
                    text_frame_count += 1

            # if it doesn't require dark frames check, but requires text frames checks, then check for text frames
            if (
                required_dark_frames is None
                and required_text_frames is not None
                and self.frame_has_text(frame)
            ):
                text_frame_count += 1

            # Check if the required dark frames and text frames are met
            respects_dark_frames = (
                required_dark_frames is None or dark_frame_count >= required_dark_frames
            )
            respects_text_frames = (
                required_text_frames is None or text_frame_count >= required_text_frames
            )
            if respects_dark_frames and respects_text_frames:
                print("\ndark frames found:", dark_frame_count, f"({dark_frame_count/self.fps:.2f} seconds)")
                print("text frames found:", text_frame_count, f"({text_frame_count/self.fps:.2f} seconds)")
                return True

            # Check if we have analyzed the required number of frames in this batch
            if frame_position > frame_limit_to_give_up:
                print(f"\n(done analyzing batch sooner - avoided processing {frame_count_to_analyze - frame_position} frames)")
                break
        
        print("\ndark frames found:", dark_frame_count, f"({dark_frame_count/self.fps:.2f} seconds)")
        print("text frames found:", text_frame_count, f"({text_frame_count/self.fps:.2f} seconds)")
        return False

    def extract_frames(
        self,
        start_frame: int,
        offset: Optional[int] = None,
        frames_to_skip: Optional[int] = None,
    ) -> Generator[MatLike, None, None]:
        """
        Extract frames from the video starting from `start_frame` and ending at `start_frame + offset` or at the end of the video if `offset` is None.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        end_frame = start_frame + offset if offset is not None else self.total_frames
        start_frame_time = self.get_time_from_frame(start_frame)
        end_frame_time = self.get_time_from_frame(end_frame)
        frame_number = copy.copy(start_frame)
        while True:
            if frames_to_skip and frame_number % frames_to_skip != 0:
                frame_number += 1
                continue

            progress = (frame_number - start_frame) / (end_frame - start_frame)

            progress_message = f"\033[92m processing frames from {start_frame} to {end_frame} (from {start_frame_time} to {end_frame_time}) \033[0m {progress:.2%}"
            print(progress_message, end="\r")

            ret, frame = self.cap.read()
            if not ret:
                break

            if offset is not None and frame_number > start_frame + offset:
                break

            frame_number += 1
            yield frame

    def write_credits_image_boundary(self, start_frame: int, end_frame: int):
        """
        Write the frames boundary of the credits detection to an image file.
        """
        start_frame_image = self.get_frame_image_from_video(start_frame)
        end_frame_image = self.get_frame_image_from_video(end_frame)
        start_time = self.get_time_from_frame(start_frame).replace(":", "-")
        end_time = self.get_time_from_frame(end_frame).replace(":", "-")
        output_directory = pathlib.Path(self.output_directory)
        start_image_file_path = (
            output_directory / f"{self.input_path.stem}_boundary_start_{start_time}.jpg"
        )
        end_image_file_path = (
            output_directory / f"{self.input_path.stem}_boundary_end_{end_time}.jpg"
        )
        cv2.imwrite(start_image_file_path, start_frame_image)
        cv2.imwrite(end_image_file_path, end_frame_image)

    def get_credits_start_frame(
        self,
        analysis_batch_seconds: int = 30,
        frames_to_skip: Optional[int] = 10,
        minutes_before_end: int = 15,
        gray_treshold: int = 20,
        required_dark_seconds: Optional[int] = None,
        required_text_seconds: Optional[int] = None,
    ) -> int:
        """
        Process the movie to check for credits every specified interval (in seconds).

        analysis_batch_seconds (int): Interval in seconds for each batch of frames that will be analysed for credits. Lower values will give a credits start time more accurate. It doesn't affect the running speed.
        frames_to_skip (float): Number of frames to skip when detecting credits. Higher values will make the process faster but less accurate. Take into account when setting "required_dark_seconds" and "required_text_frames" and "analysis_batch_seconds".
        minutes_before_end (int): Number of minutes before the end of the video to start the analysis.
        gray_treshold (int): Threshold value to determine if a frame is dark.
        required_dark_seconds (int): number of seconds (with frames_to_skip as an interval) with text to consider a batch of frames as including the credits. This depends on the movie FPS, the "frames_to_skip" parameter and their relationship.
        required_dark_seconds (int): number of seconds (with frames_to_skip as an interval) with dark frames to consider it as credits. This depends on the movie FPS, the "frames_to_skip" parameter and their relationship.
        """

        # validate that analysis_batch_seconds is bigger than seconds_to_skip
        seconds_to_skip = frames_to_skip / self.fps
        frame_count_to_analyze_per_batch = analysis_batch_seconds / seconds_to_skip

        if required_dark_seconds and required_text_seconds and required_text_seconds > required_dark_seconds:
            raise ValueError(
                "required_text_seconds must be less or equal than required_dark_seconds if both are provided"
            )
        if analysis_batch_seconds <= seconds_to_skip:
            raise ValueError(
                "analysis_batch_seconds must be greater than seconds_to_skip"
            )

        # validate that (analysis_batch_seconds / seconds_to_skip) is greater than required_dark_seconds
        if (
            required_dark_seconds
            and seconds_to_skip
            and (frame_count_to_analyze_per_batch) < required_dark_seconds
        ):
            raise ValueError(
                "analysis_batch_seconds / seconds_to_skip must be greater than required_dark_seconds"
            )

        # validate that (analysis_batch_seconds / seconds_to_skip) is greater than required_text_seconds
        if (
            required_text_seconds
            and seconds_to_skip
            and (frame_count_to_analyze_per_batch) < required_text_seconds
        ):
            raise ValueError(
                "analysis_batch_seconds / seconds_to_skip must be greater than required_text_seconds"
            )

        analysis_start_frame = max(
            0, self.total_frames - int(minutes_before_end * 60 * self.fps)
        )
        print(f"Analyzing the last {minutes_before_end} minutes of the video")
        analysis_start_frame_time = self.get_time_from_frame(analysis_start_frame)
        print(
            f"Analysis Start frame: {analysis_start_frame} ({analysis_start_frame_time})\n"
        )

        # Convert seconds to frame count
        batch_frame_count = int(self.fps * analysis_batch_seconds)
        start_frame = analysis_start_frame

        while True:
            end_frame = start_frame + batch_frame_count
            frames = self.extract_frames(
                start_frame, batch_frame_count, frames_to_skip=frames_to_skip
            )

            # Check if this set of frames have credits
            if self.frames_have_credits(
                frames=frames,
                frame_count_to_analyze=frame_count_to_analyze_per_batch,
                gray_treshold=gray_treshold,
                required_dark_seconds=required_dark_seconds,
                required_text_seconds=required_text_seconds,
            ):
                start_frame_time = self.get_time_from_frame(start_frame)
                end_frame_time = self.get_time_from_frame(end_frame)
                print(
                    f"\n\n GREAT SUCCESS! Credits detected between frame {start_frame} and {end_frame} ({start_frame_time} and {end_frame_time})"
                )
                self.write_credits_image_boundary(start_frame, end_frame)
                return start_frame

            print(
                f"\nNo credits detected in frames {start_frame} to {end_frame} - moving to next interval.\n\n"
            )
            # Check if we have reached the end of the video
            if end_frame >= self.total_frames:
                print("\nEnd of video reached")
                raise CreditsNotFoundError(
                    f"Credits not detected in the last {minutes_before_end} minutes of the video."
                )

            # Move to the next interval
            start_frame += batch_frame_count

    def export_credits_video(
        self,
        start_frame: Optional[int] = None,
        start_seconds: Optional[int] = None,
        start_time: Optional[str] = None,  # "02:32:44"
        initial_buffer_seconds: int = 0,
        output_path: Optional[Union[str, pathlib.Path]] = None,
    ):
        """
        Extract credit frames from credit_start_frame_number to the end of the video
        Write credit frames to output video

        start_frame (int): Frame number to start extracting credits from.
        start_seconds (int): Seconds to start extracting credits from - will be used to calculate the start frame if provided.
        start_time (str): Time in the format "HH:MM:SS" to start extracting credits from - will be used to calculate the start frame if provided.
        initial_buffer_seconds (int): Number of seconds to add to the start frame to include some initial buffer before the credits start - safety measure to ensure the credits are not cut off.
        output_path (Union[str, pathlib.Path]): Path to save the output video file. If not provided, the output will be saved in the same directory as the input video with "_credits.mp4" appended to the filename.
        
        """

        if output_path is None:
            output_path = (
                pathlib.Path(self.output_directory)
                / f"{self.input_path.stem}_credits.mp4"
            )

        if start_seconds is not None:
            start_frame = int(start_seconds * self.fps)

        elif start_time is not None:
            start_frame = int(
                self.fps
                * sum(
                    int(x) * 60**i
                    for i, x in enumerate(reversed(start_time.split(":")))
                )
            )
        elif start_frame is None:
            raise ValueError(
                "One of start_frame, start_seconds or start_time must be provided."
            )
        
        start_frame -= initial_buffer_seconds * self.fps
        start_time = self.get_time_from_frame(start_frame)

        print(f"\n\nExporting credits video from {start_frame} ({start_time})...")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        credit_frames = self.extract_frames(start_frame)

        # Create VideoWriter for output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        for frame in credit_frames:
            out.write(frame)

        out.release()

        print(f"\nCredit video saved to {output_path}")
