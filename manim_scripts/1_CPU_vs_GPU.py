from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService

class Introduction(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        GTTSService()
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU programming", font_size=72).shift(2*UP)
    with self.voiceover(text="Hello and welcome to episode 1 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("GPU vs CPU", font_size=48).next_to(title, DOWN)
    desc = BulletedList("Architectural Differences", "Latency and Throughput", "When is it beneficial to use a GPU", font_size=32).next_to(subtitle, DOWN)

    with self.voiceover(text="In this episode we are going to discuss the key differences between the gpu and the cpu") as trk:
      self.play(Write(subtitle))

    with self.voiceover(text="How the architecture of the two differs") as trk:
      self.play(Write(desc[0]))

    with self.voiceover(text="What is this latency and throughput stuff that is always mentioned when talking about those things") as trk:
      self.play(Write(desc[1]))

    with self.voiceover(text="and when to use one over the other") as trk:
      self.play(Write(desc[2]))

    with self.voiceover(text="And finally, we are going to crack open the editor and write some code") as trk:
      for i in range(3):
        self.play(Unwrite(desc[2-i]), run_time=trk.duration/5)
      self.play(Unwrite(subtitle), run_time=trk.duration/5)
      self.play(Unwrite(title), run_time=trk.duration/5)


