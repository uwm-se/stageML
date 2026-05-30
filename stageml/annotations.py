

from __future__ import annotations
from dataclasses import dataclass
#from importlib.metadata.diagnose import inspect
from typing import Callable, Any
import inspect


@dataclass(frozen=True)
class BindingTime:
    level: int          # 0 = static, 1 = dynamic
    name:  str

    def __repr__(self):
        return self.name

    def join(self, other: "BindingTime") -> "BindingTime":
        
        return stage1 if (self.level == 1 or other.level == 1) else stage0


stage0 = BindingTime(level=0, name="stage0")   # compile-time static
stage1 = BindingTime(level=1, name="stage1")   # runtime dynamic




def build_staging_env(fn: Callable) -> dict[str, BindingTime]:

    hints = fn.__annotations__
    gamma = {}
    for name, annotation in hints.items():
        if name == "return":
            continue
        if isinstance(annotation, BindingTime):
            gamma[name] = annotation
        else:
            gamma[name] = stage1
    for name in inspect.signature(fn).parameters:
        if name not in gamma:
            gamma[name] = stage1
    return gamma



def compile_staged(fn: Callable) -> Callable:
  
    gamma = build_staging_env(fn)
    fn._gamma        = gamma
    fn._staged       = True
    fn._compiled_fn  = None

    def analyze():
      
        print(f"\n{'─'*50}")
        print(f"StageML Staging Analysis: {fn.__name__}")
        print(f"{'─'*50}")
        print(f"{'Parameter':<20} {'Binding Time':<15} {'Meaning'}")
        print(f"{'─'*50}")
        for name, bt in gamma.items():
            meaning = "compile-time static" if bt == stage0 else "runtime dynamic"
            print(f"  {name:<18} {str(bt):<15} {meaning}")
        print(f"{'─'*50}")

    def compile():
       
        print(f"[StageML] Compiling {fn.__name__}...")
        print(f"  Staging environment: {gamma}")
        return fn

    fn.analyze  = analyze
    fn.compile  = compile
    return fn
