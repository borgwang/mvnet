import datetime
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
    #print(f"`{self.name}` time_cost={datetime.timedelta(seconds=self.__seconds)}")

  def reset(self) -> None:
    self.__seconds = 0.0
