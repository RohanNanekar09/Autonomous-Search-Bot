#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        self.bridge = CvBridge()
        self.model  = YOLO("yolov8n.pt")

        # ── Timers ────────────────────────────────────────────────────────────
        self.last_inference = self.get_clock().now()   # run inference every 1s
        self.last_log       = self.get_clock().now()   # print to terminal every 5s

        self.inference_interval = Duration(seconds=1)
        self.log_interval       = Duration(seconds=5)

        # ── Subscriber ────────────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            "/yolo/detections",
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            "/yolo/image_annotated",
            10
        )

        self.get_logger().info("YOLO node started | inference: 1s | log: 5s")

    def image_callback(self, msg):

        now = self.get_clock().now()

        # ── Only run inference every 1 second ─────────────────────────────────
        if (now - self.last_inference) < self.inference_interval:
            return

        self.last_inference = now

        # ── Convert image ─────────────────────────────────────────────────────
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # ── Run YOLO (verbose=False silences YOLO internal logs) ──────────────
        results = self.model(frame, verbose=False)

        # ── Build Detection2DArray (always publish even if empty) ─────────────
        detection_array        = Detection2DArray()
        detection_array.header = msg.header

        should_log = (now - self.last_log) >= self.log_interval

        if len(results[0].boxes) == 0:
            # Always publish so waypoint node keeps receiving at 1s rate
            self.detection_pub.publish(detection_array)

            if should_log:
                self.get_logger().info("No objects detected")
                self.last_log = now
            return

        # ── Pack detections ───────────────────────────────────────────────────
        log_lines = []

        for box in results[0].boxes:
            cls        = int(box.cls)
            name       = self.model.names[cls]
            confidence = float(box.conf)
            x_center   = float(box.xywh[0][0])
            y_center   = float(box.xywh[0][1])
            width      = float(box.xywh[0][2])
            height     = float(box.xywh[0][3])

            det                        = Detection2D()
            det.header                 = msg.header
            det.bbox.center.position.x = x_center
            det.bbox.center.position.y = y_center
            det.bbox.size_x            = width
            det.bbox.size_y            = height

            hyp                     = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score    = confidence
            det.results.append(hyp)

            detection_array.detections.append(det)

            log_lines.append(
                f"  [{name}] conf: {confidence:.2f} "
                f"center: ({x_center:.1f}, {y_center:.1f})"
            )

        # ── Publish detections every 1 second ─────────────────────────────────
        self.detection_pub.publish(detection_array)

        # ── Log only every 5 seconds ──────────────────────────────────────────
        if should_log:
            self.get_logger().info(
                f"Detected {len(log_lines)} object(s):\n" + "\n".join(log_lines)
            )
            self.last_log = now

        # ── Publish annotated image for RViz ──────────────────────────────────
        annotated_frame      = results[0].plot()
        annotated_msg        = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
        annotated_msg.header = msg.header
        self.image_pub.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()