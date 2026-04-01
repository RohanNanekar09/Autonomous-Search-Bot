#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import subprocess
import threading
import queue
import time
import os
from faster_whisper import WhisperModel


class VoiceNode(Node):

    def __init__(self):
        super().__init__('voice_node')

        # ── State ─────────────────────────────────────────────────────────────
        self.listen_enabled    = True   # listen for "find" at startup
        self.is_speaking       = False
        self.tts_queue         = queue.Queue()

        # ── ROS interfaces ────────────────────────────────────────────────────
        self.voice_pub = self.create_publisher(String, 'voice_cmd', 10)

        self.create_subscription(
            String, 'speak', self.speak_callback, 10)

        self.create_subscription(
            Bool, '/listen_enable', self.listen_enable_callback, 10)

        # ── Load Whisper ──────────────────────────────────────────────────────
        self.get_logger().info("Loading Whisper model...")
        self.model = WhisperModel("base", device="cpu", compute_type="int8")
        self.get_logger().info("Voice node ready. Say: find <object>")

        # ── Start threads ─────────────────────────────────────────────────────
        threading.Thread(target=self.tts_loop,    daemon=True).start()
        threading.Thread(target=self.listen_loop, daemon=True).start()

    # ── Enable/disable listening from waypoint ────────────────────────────────
    def listen_enable_callback(self, msg):
        self.listen_enabled = msg.data
        self.get_logger().info(
            f"Listening {'ENABLED' if msg.data else 'DISABLED'}")

    # ── TTS callback — queues text ────────────────────────────────────────────
    def speak_callback(self, msg):
        self.tts_queue.put(msg.data)

    # ── Speak helper — called internally ─────────────────────────────────────
    def speak(self, text):
        self.tts_queue.put(text)

    # ── TTS loop ──────────────────────────────────────────────────────────────
    def tts_loop(self):
        while rclpy.ok():
            try:
                text = self.tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            text = text.replace('"', '').replace("'", '')
            if not text:
                continue

            self.get_logger().info(f"Speaking: {text}")
            self.is_speaking = True
            subprocess.run(
                ["espeak", "-s", "150", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.0)   # buffer after speaking to avoid echo
            self.is_speaking = False

    # ── Keyword correction ────────────────────────────────────────────────────
    def correct_text(self, text):
        replacements = {
            "fine ":         "find ",
            "find find ":    "find ",
            "fan ":          "find ",
            "search ":       "find ",
            "fine, ":        "find ",
            "drunk":         "truck",
            "drug":          "truck",
            "drop":          "truck",
            "rock":          "truck",
            "share":         "chair",
            "char ":         "chair ",
            "cheer":         "chair",
            "fridge":        "refrigerator",
            "refridgerator": "refrigerator",
            "freezer":       "refrigerator",
            "refrigerate":   "refrigerator",
            "hydrant":       "fire hydrant",
            "fire hydran":   "fire hydrant",
          
            # ── come back corrections ──────────────────────────────────────
            "come back":     "come back",
            "comeback":      "come back",
            "go back":       "come back",
            "go bag":        "come back",
            "come bag":      "come back",
            "calm back":     "come back",
            "calm bag":      "come back",
            "return":        "come back",
            "go to start":   "come back",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    # ── Listen loop ───────────────────────────────────────────────────────────
    def listen_loop(self):
        tmp_file = "/tmp/voice_input.wav"

        while rclpy.ok():

            # ── Gate: only listen when enabled and not speaking ───────────────
            if not self.listen_enabled or self.is_speaking:
                time.sleep(0.1)
                continue

            self.get_logger().info("Listening...")

            # ── Record using PulseAudio default (same as working arecord) ───
            subprocess.run(
                ["arecord", "-f", "S16_LE", "-r", "16000",
                 "-d", "2", "-q", tmp_file],
                stderr=subprocess.DEVNULL
            )

            if not os.path.exists(tmp_file):
                continue

            # ── Skip if speaking started during recording ─────────────────────
            if self.is_speaking:
                os.unlink(tmp_file)
                continue

            # ── Convert stereo to mono for Whisper ───────────────────────────
            mono_file = "/tmp/voice_mono.wav"
            subprocess.run(
                ["sox", tmp_file, mono_file, "remix", "1"],
                stderr=subprocess.DEVNULL
            )
            transcribe_file = mono_file if os.path.exists(mono_file) else tmp_file

            # ── Transcribe with Whisper ───────────────────────────────────────
            try:
                segments, _ = self.model.transcribe(
                    transcribe_file,
                    language="en",
                    vad_filter=False
                )
                text = " ".join([s.text for s in segments]).strip().lower()
            except Exception as e:
                self.get_logger().error(f"Transcription error: {e}")
                text = ""
            finally:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
                if os.path.exists(mono_file):
                    os.unlink(mono_file)

            if not text:
                continue

            # ── Apply corrections ─────────────────────────────────────────────
            text = self.correct_text(text)
            self.get_logger().info(f"Heard: {text}")

            # ── Handle "find <object>" — must have object after find ──────────
            if text.startswith("find "):
                obj = text.replace("find ", "").strip()
                if not obj:
                    # "find" alone — not valid, keep listening
                    self.get_logger().info("Ignored: no object specified after find")
                    continue
                msg = String()
                msg.data = text
                self.voice_pub.publish(msg)
                self.speak(f"Finding {obj}")
                self.get_logger().info("Command published")
                self.listen_enabled = False
                continue

            # ── Handle "come back" response ───────────────────────────────────
            if "come back" in text:
                msg = String()
                msg.data = text
                self.voice_pub.publish(msg)
                self.get_logger().info("Response published: come back")
                self.listen_enabled = False
                continue

            # ── Anything else — ignore, keep listening ────────────────────────
            self.get_logger().info(f"Ignored: not a valid command ({text})")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()