import pathlib
import time
from typing import Generator, Optional, Union
import cv2
from cv2.typing import MatLike
import numpy as np
import pytesseract


class MovieCreditsExtractor:
    def __init__(
        self,
        input_path: Union[str, pathlib.Path],
        output_path: Union[str, pathlib.Path] = "credits.mp4",
        analysis_batch_seconds: int = 30,
        frames_to_skip: Optional[int] = 10,
        minutes_before_end: int = 15,
        gray_treshold: int = 20,
        required_dark_frames: Optional[int] = None,
        required_text_frames: Optional[int] = None,
    ):
        """
        Initialize the MovieCreditsExtractor object with the input video path and other parameters.
        input_path (Union[str,pathlib.Path]): Path to the input MP4 video file.
        output_path (Union[str,pathlib.Path]): Path to save the output video with credits.
        analysis_batch_seconds (int): Interval in seconds for each batch of frames that will be analysed for credits. Lower values will give a credits start time more accurate. It doesn't affect the running speed.
        frames_to_skip (int): Number of frames to skip when detecting credits. Higher values will make the process faster but less accurate. Take into account when setting "required_dark_frames" and "required_text_frames" and "analysis_batch_seconds".
        minutes_before_end (int): Number of minutes before the end of the video to start the analysis.
        gray_treshold (int): Threshold value to determine if a frame is dark.
        required_dark_frames (int): Number of consecutive frames (with frames_to_skip as an interval) with dark background to consider it as credits. This depends on the movie FPS, the "frames_to_skip" parameter and their relationship.
        required_text_frames (int): Number of consecutive frames (with frames_to_skip as an interval) with text to consider it as credits. This depends on the movie FPS, the "frames_to_skip" parameter and their relationship.
        """
        self.input_path = input_path
        self.analysis_batch_seconds = analysis_batch_seconds
        self.frames_to_skip = frames_to_skip
        self.minutes_before_end = minutes_before_end
        self.gray_treshold = gray_treshold
        self.required_dark_frames = required_dark_frames
        self.required_text_frames = required_text_frames
        self.output_path = output_path

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
            Duration Display: {duration_display}
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

    def frame_is_dark(self, frame: MatLike) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        return (
            mean_intensity < self.gray_treshold
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
        frames: Generator[MatLike, None, None],
    ) -> bool:
        """
        Function to detect if a given set of frames have movie credits or not.
        Count the number of dark frames and text frames in the given set of frames.
        Check if the number of dark frames and text frames meet the required criteria.
        """
        dark_frame_count = text_frame_count = 0
        for frame_position, frame in enumerate(frames):
            if self.frames_to_skip and frame_position % self.frames_to_skip != 0:
                # Skip frames to speed up processing
                continue
            if self.required_dark_frames is not None and self.frame_is_dark(frame):
                dark_frame_count += 1
            if self.required_text_frames is not None and self.frame_has_text(frame):
                text_frame_count += 1

        respects_dark_frames = (
            self.required_dark_frames is None
            or dark_frame_count >= self.required_dark_frames
        )
        respects_text_frames = (
            self.required_text_frames is None
            or text_frame_count >= self.required_text_frames
        )

        return respects_dark_frames and respects_text_frames

    def extract_frames(
        self, start_frame: int, offset: Optional[int] = None
    ) -> Generator[MatLike, None, None]:
        """
        Extract frames from the video starting from `start_frame` and ending at `start_frame + offset` or at the end of the video if `offset` is None.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        end_frame = start_frame + offset if offset is not None else self.total_frames
        start_frame_time = self.get_time_from_frame(start_frame)
        end_frame_time = self.get_time_from_frame(end_frame)
        print("\n")
        while True:
            progress = (self.cap.get(cv2.CAP_PROP_POS_FRAMES) - start_frame) / (
                end_frame - start_frame
            )

            progress_message = f"processing frames from {start_frame} to {end_frame} (from {start_frame_time} to {end_frame_time}) {progress:.2%}"
            print(progress_message, end="\r")

            ret, frame = self.cap.read()
            if not ret:
                break

            current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if offset is not None and current_frame_number > start_frame + offset:
                break
            yield frame

    def write_credits_image_boundary(self, start_frame: int, end_frame: int):
        """
        Write the frames boundary of the credits detection to an image file.
        """
        start_frame_image = self.get_frame_image_from_video(start_frame)
        end_frame_image = self.get_frame_image_from_video(end_frame)
        start_time = self.get_time_from_frame(start_frame).replace(":", "-")
        end_time = self.get_time_from_frame(end_frame).replace(":", "-")
        _output_path = pathlib.Path(self.output_path)
        start_image_file_path = (
            _output_path.parent / f"{_output_path.stem}_boundary_start_{start_time}.jpg"
        )
        end_image_file_path = (
            _output_path.parent / f"{_output_path.stem}_boundary_end_{end_time}.jpg"
        )
        cv2.imwrite(start_image_file_path, start_frame_image)
        cv2.imwrite(end_image_file_path, end_frame_image)

    def get_credits_start_frame(self) -> int:
        """
        Process the movie to check for credits every specified interval (in seconds).
        """

        analysis_start_frame = max(
            0, self.total_frames - int(self.minutes_before_end * 60 * self.fps)
        )
        print(f"Analyzing the last {self.minutes_before_end} minutes of the video")
        analysis_start_frame_time = self.get_time_from_frame(analysis_start_frame)
        print(
            f"Analysis Start frame: {analysis_start_frame} ({analysis_start_frame_time})"
        )

        # Convert seconds to frame count
        batch_frame_count = int(self.fps * self.analysis_batch_seconds)

        start_frame = analysis_start_frame
        while True:
            end_frame = start_frame + batch_frame_count
            frames = self.extract_frames(start_frame, batch_frame_count)

            # Check if this set of frames have credits
            if self.frames_have_credits(frames):
                start_frame_time = self.get_time_from_frame(start_frame)
                end_frame_time = self.get_time_from_frame(end_frame)
                print(
                    f"\n\n GREAT SUCCESS! Credits detected between frame {start_frame} and {end_frame} ({start_frame_time} and {end_frame_time})"
                )
                self.write_credits_image_boundary(start_frame, end_frame)
                return start_frame

            print(
                f"\nNo credits detected in frames {start_frame} to {end_frame} - moving to next interval."
            )
            # Check if we have reached the end of the video
            if end_frame >= self.total_frames:
                print("\nEnd of video reached")
                raise ValueError(
                    f"Credits not detected in the last {self.minutes_before_end} minutes of the video."
                )

            # Move to the next interval
            start_frame += batch_frame_count

    def export_credits_video(
        self,
        start_frame: Optional[int] = None,
        start_seconds: Optional[int] = None,
        start_time: Optional[str] = None,  # "02:32:44"
    ):
        # Extract credit frames from credit_start_frame_number to the end of the video
        # Write credit frames to output video
        if start_seconds is not None:
            start_frame = int(start_seconds * self.fps)
            start_time = self.get_time_from_frame(start_frame)

        elif start_time is not None:
            start_frame = int(
                self.fps
                * sum(
                    int(x) * 60**i
                    for i, x in enumerate(reversed(start_time.split(":")))
                )
            )
        elif start_frame is not None:
            start_time = self.get_time_from_frame(start_frame)
        else:
            raise ValueError(
                "One of start_frame, start_seconds or start_time must be provided."
            )
        print(f"\n\nExporting credits video from {start_frame} ({start_time})...")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        credit_frames = self.extract_frames(start_frame)

        # Create VideoWriter for output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        for frame in credit_frames:
            out.write(frame)

        out.release()

        print(f"\nCredit video saved to {self.output_path}")

        
