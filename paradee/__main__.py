"""python -m paradee "Hello there." -o hello.wav"""
import argparse, sys, wave
import numpy as np
from . import Paradee, SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser(prog="paradee", description="Speak text with Paradee and save a WAV file.")
    ap.add_argument("text", nargs="?", help="text to speak (read from stdin when omitted)")
    ap.add_argument("-o", "--out", default="paradee.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fp32", action="store_true", help="use the fp32 model instead of int8")
    ap.add_argument("--model-version", default="1.1", choices=["1.1", "1.0"])
    ap.add_argument("--model-path", default=None, help="path to local ONNX model file")
    ap.add_argument("--config-path", default=None, help="path to local config.json file")
    a = ap.parse_args()
    text = a.text if a.text is not None else sys.stdin.read()
    tts = Paradee(
        version=a.model_version,
        quantized=not a.fp32,
        model_path=a.model_path,
        config_path=a.config_path,
    )
    audio = tts(text, speed=a.speed)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(pcm.tobytes())
    print(f"wrote {a.out} ({len(audio) / SAMPLE_RATE:.1f} s)")


if __name__ == "__main__":
    main()
