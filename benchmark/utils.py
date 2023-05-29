import time


class Timer:
  def __init__(self, name:str):
    self.name = name
    self.__seconds = 0.0

  @property
  def sec(self) -> float:
    return self.__seconds

  @property
  def ms(self) -> float:
    return self.__seconds * 1000

  def __enter__(self):
    self.st = time.monotonic()
    return self

  def __exit__(self, *args, **kwargs):
    self.__seconds += time.monotonic() - self.st

  def reset(self) -> None:
    self.__seconds = 0.0

FS = {
  "UNDERLINE": "\033[4m",
  "BLUE": "\033[94m",
  "HIGHLIGHT": "\x1b[6;30;42m",  # highlight green
  "HIGHLIGHT_RED": "\x1b[6;30;41m",  # highlight red
  "DARKGREY": "\033[90m",
  "LIGHTGREY": "\033[37m",
  "RED": "\033[91m",
  "YELLOW": "\033[93m",
  "CYAN": "\033[96m",
  "GREEN": "\033[92m",
  "BOLD": "\033[1m",
  "WHITE": "\033[1;37m",
  "STRIKETHROUGH": "\033[9m",
  "ENDC": "\033[0m"
}
