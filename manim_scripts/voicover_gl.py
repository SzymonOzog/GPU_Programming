from math import ceil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator, List
import re
import typing as t

from manim_voiceover.services.base import SpeechService
from manim_voiceover.tracker import VoiceoverTracker
from manim_voiceover.helper import chunks, remove_bookmarks

import numpy as np
from manim import logger

from scipy.interpolate import interp1d

from manimlib import Scene
from manimlib.config import manim_config
from manim_voiceover.modify_audio import get_duration

AUDIO_OFFSET_RESOLUTION = 10_000_000


class TimeInterpolator:
    def __init__(self, word_boundaries: List[dict]):
        self.x = []
        self.y = []
        for wb in word_boundaries:
            self.x.append(wb["text_offset"])
            self.y.append(wb["audio_offset"] / AUDIO_OFFSET_RESOLUTION)

        self.f = interp1d(self.x, self.y)

    def interpolate(self, distance: int) -> np.ndarray:
        try:
            return self.f(distance)
        except:
            logger.warning(
                "TimeInterpolator received weird input, there may be something wrong with the word boundaries."
            )
            return self.y[-1]


class VoiceoverTracker:
    """Class to track the progress of a voiceover in a scene."""

    def __init__(self, scene: Scene, data: dict, cache_dir: str, dummy: bool = False):
        """Initializes a VoiceoverTracker object.

        Args:
            scene (Scene): The scene to which the voiceover belongs.
            path (str): The path to the JSON file containing the voiceover data.
        """
        self.scene = scene
        self.data = data
        self.cache_dir = cache_dir
        self.duration = 0.01 if dummy else get_duration(Path(cache_dir) / self.data["final_audio"])
        last_t = scene.time
        # last_t = scene.renderer.time
        if last_t is None:
            last_t = 0
        self.start_t = last_t
        self.end_t = last_t + self.duration

        if "word_boundaries" in self.data:
            self._process_bookmarks()

    def _get_fallback_word_boundaries(self):
        """
        Returns dummy word boundaries assuming a linear mapping between
        text and audio. Used when word boundaries are not available.
        """
        input_text = remove_bookmarks(self.data["input_text"])
        return [
            {
                "audio_offset": 0,
                "text_offset": 0,
                "word_length": len(input_text),
                "text": self.data["input_text"],
                "boundary_type": "Word",
            },
            {
                "audio_offset": self.duration * AUDIO_OFFSET_RESOLUTION,
                "text_offset": len(input_text),
                "word_length": 1,
                "text": ".",
                "boundary_type": "Word",
            },
        ]

    def _process_bookmarks(self) -> None:
        self.bookmark_times = {}
        self.bookmark_distances = {}

        word_boundaries = self.data["word_boundaries"]
        if not word_boundaries or len(word_boundaries) < 2:
            logger.warning(
                f"Word boundaries for voiceover {self.data['input_text']} are not "
                "available or are insufficient. Using fallback word boundaries."
            )
            word_boundaries = self._get_fallback_word_boundaries()

        self.time_interpolator = TimeInterpolator(word_boundaries)

        net_text_len = len(remove_bookmarks(self.data["input_text"]))
        if "transcribed_text" in self.data:
            transcribed_text_len = len(self.data["transcribed_text"].strip())
        else:
            transcribed_text_len = net_text_len

        self.input_text = self.data["input_text"]
        self.content = ""

        # Mark bookmark distances
        # parts = re.split("(<bookmark .*/>)", self.input_text)
        parts = re.split(r"(<bookmark\s*mark\s*=[\'\"]\w*[\"\']\s*/>)", self.input_text)
        for p in parts:
            matched = re.match(r"<bookmark\s*mark\s*=[\'\"](.*)[\"\']\s*/>", p)
            if matched:
                self.bookmark_distances[matched.group(1)] = len(self.content)
            else:
                self.content += p

        for mark, dist in self.bookmark_distances.items():
            # Normalize text offset
            elapsed = self.time_interpolator.interpolate(
                dist * transcribed_text_len / net_text_len
            )
            self.bookmark_times[mark] = self.start_t + elapsed

    def get_remaining_duration(self, buff: float = 0.0) -> float:
        """Returns the remaining duration of the voiceover.

        Args:
            buff (float, optional): A buffer to add to the remaining duration. Defaults to 0.

        Returns:
            int: The remaining duration of the voiceover in seconds.
        """
        # result= max(self.end_t - self.scene.last_t, 0)
        result = max(self.end_t - self.scene.time + buff, 0)
        # print(result)
        return result

    def _check_bookmarks(self):
        if not hasattr(self, "bookmark_times"):
            raise Exception(
                "Word boundaries are required for timing with bookmarks. "
                "Manim Voiceover currently supports auto-transcription using OpenAI Whisper, "
                "but this is not enabled for each speech service by default. "
                "You can enable it by setting transcription_model='base' in your speech service initialization. "
                "If the performance of the base model is not satisfactory, you can use one of the larger models. "
                "See https://github.com/openai/whisper for a list of all the available models."
            )

    def time_until_bookmark(
        self, mark: str, buff: int = 0, limit: Optional[int] = None
    ) -> int:
        """Returns the time until a bookmark.

        Args:
            mark (str): The `mark` attribute of the bookmark to count up to.
            buff (int, optional): A buffer to add to the remaining duration, in seconds. Defaults to 0.
            limit (Optional[int], optional): A maximum value to return. Defaults to None.

        Returns:
            int:
        """
        self._check_bookmarks()
        if not mark in self.bookmark_times:
            raise Exception("There is no <bookmark mark='%s' />" % mark)
        result = max(self.bookmark_times[mark] - self.scene.time + buff, 0)
        if limit is not None:
            result = min(limit, result)
        return result


# SCRIPT_FILE_PATH = "media/script.txt"


class VoiceoverScene(Scene):
    """A scene class that can be used to add voiceover to a scene."""

    speech_service: SpeechService
    current_tracker: Optional[VoiceoverTracker]
    create_subcaption: bool
    create_script: bool
    voiceovers_in_embed: bool = False
    mock: bool = False

    def set_speech_service(
        self,
        speech_service: SpeechService,
        create_subcaption: bool = False,
    ) -> None:
        """Sets the speech service to be used for the voiceover. This method
        should be called before adding any voiceover to the scene.

        Args:
            speech_service (SpeechService): The speech service to be used.
            create_subcaption (bool, optional): Whether to create subcaptions for the scene. Defaults to True. If `config.save_last_frame` is True, the argument is
            ignored and no subcaptions will be created.
        """
        self.speech_service = speech_service
        self.current_tracker = None
        # TODO not supported
        self.create_subcaption = False
        self.timestamps = []

    def add_voiceover_text(
        self,
        text: str,
        subcaption: Optional[str] = None,
        max_subcaption_len: int = 70,
        subcaption_buff: float = 0.1,
        **kwargs,
    ) -> VoiceoverTracker:
        """Adds voiceover to the scene.

        Args:
            text (str): The text to be spoken.
            subcaption (Optional[str], optional): Alternative subcaption text. If not specified, `text` is chosen as the subcaption. Defaults to None.
            max_subcaption_len (int, optional): Maximum number of characters for a subcaption. Subcaptions that are longer are split into chunks that are smaller than `max_subcaption_len`. Defaults to 70.
            subcaption_buff (float, optional): The duration between split subcaption chunks in seconds. Defaults to 0.1.

        Returns:
            VoiceoverTracker: The tracker object for the voiceover.
        """
        if not hasattr(self, "speech_service"):
            raise Exception(
                "You need to call init_voiceover() before adding a voiceover."
            )

        dict_ = self.speech_service._wrap_generate_from_text(text, **kwargs)
        tracker = VoiceoverTracker(self, dict_, self.speech_service.cache_dir)
        self.add_sound(str(Path(self.speech_service.cache_dir) / dict_["final_audio"]))
        self.current_tracker = tracker

        # if self.create_script:
        #     self.save_to_script_file(text)

        if self.create_subcaption:
            if subcaption is None:
                subcaption = remove_bookmarks(text)

            self.add_wrapped_subcaption(
                subcaption,
                tracker.duration,
                subcaption_buff=subcaption_buff,
                max_subcaption_len=max_subcaption_len,
            )

        return tracker

    def add_wrapped_subcaption(
        self,
        subcaption: str,
        duration: float,
        subcaption_buff: float = 0.1,
        max_subcaption_len: int = 70,
    ) -> None:
        """Adds a subcaption to the scene. If the subcaption is longer than `max_subcaption_len`, it is split into chunks that are smaller than `max_subcaption_len`.

        Args:
            subcaption (str): The subcaption text.
            duration (float): The duration of the subcaption in seconds.
            max_subcaption_len (int, optional): Maximum number of characters for a subcaption. Subcaptions that are longer are split into chunks that are smaller than `max_subcaption_len`. Defaults to 70.
            subcaption_buff (float, optional): The duration between split subcaption chunks in seconds. Defaults to 0.1.
        """
        subcaption = " ".join(subcaption.split())
        n_chunk = ceil(len(subcaption) / max_subcaption_len)
        tokens = subcaption.split(" ")
        chunk_len = ceil(len(tokens) / n_chunk)
        chunks_ = list(chunks(tokens, chunk_len))
        try:
            assert len(chunks_) == n_chunk or len(chunks_) == n_chunk - 1
        except AssertionError:
            import ipdb

            ipdb.set_trace()

        subcaptions = [" ".join(i) for i in chunks_]
        subcaption_weights = [
            len(subcaption) / len("".join(subcaptions)) for subcaption in subcaptions
        ]

        current_offset = 0
        for idx, subcaption in enumerate(subcaptions):
            chunk_duration = duration * subcaption_weights[idx]
            self.add_subcaption(
                subcaption,
                duration=max(chunk_duration - subcaption_buff, 0),
                offset=current_offset,
            )
            current_offset += chunk_duration

    def add_voiceover_ssml(self, ssml: str, **kwargs) -> None:
        raise NotImplementedError("SSML input not implemented yet.")

    # def save_to_script_file(self, text: str) -> None:
    #     text = " ".join(text.split())
    #     # script_file_path = Path(config.get_dir("output_file")).with_suffix(".script.srt")
    #     with open(SCRIPT_FILE_PATH, "a") as f:
    #         f.write(text)
    #         f.write("\n\n")

    def wait_for_voiceover(self) -> None:
        """Waits for the voiceover to finish."""
        if not hasattr(self, "current_tracker"):
            return
        if self.current_tracker is None:
            return

        self.safe_wait(self.current_tracker.get_remaining_duration())

    def safe_wait(self, duration: float) -> None:
        """Waits for a given duration. If the duration is less than one frame, it waits for one frame.

        Args:
            duration (float): The duration to wait for in seconds.
        """
        fps = manim_config["camera"]["fps"]
        if duration > 1 / 30:
            self.wait(duration)

    def wait_until_bookmark(self, mark: str) -> None:
        """Waits until a bookmark is reached.

        Args:
            mark (str): The `mark` attribute of the bookmark to wait for.
        """
        if self.current_tracker is None or self.mock:
            return
        self.safe_wait(self.current_tracker.time_until_bookmark(mark))

    @contextmanager
    def voiceover(
        self, text: t.Optional[str] = None, ssml: t.Optional[str] = None, **kwargs
    ) -> Generator[VoiceoverTracker, None, None]:
        """The main function to be used for adding voiceover to a scene.

        Args:
            text (str, optional): The text to be spoken. Defaults to None.
            ssml (str, optional): The SSML to be spoken. Defaults to None.

        Yields:
            Generator[VoiceoverTracker, None, None]: The voiceover tracker object.
        """
        if self.mock:
            text = "pass"
        start_time = self.time
        if text is None and ssml is None:
            raise ValueError("Please specify either a voiceover text or SSML string.")

        try:
            if self.window is not None and not self.voiceovers_in_embed:
                yield VoiceoverTracker(self, "", None, True)
            elif text is not None:
                yield self.add_voiceover_text(text, **kwargs)
            elif ssml is not None:
                yield self.add_voiceover_ssml(ssml, **kwargs)
        finally:
            self.wait_for_voiceover()
            self.timestamps.append(f"{start_time},{self.time}")

    def print_timestamps(self):
        print(";".join(self.timestamps))
