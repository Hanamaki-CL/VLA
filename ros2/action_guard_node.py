#!/usr/bin/env python3
"""ROS 2 dry-run guard. Never connects to robot hardware or publishes motor commands.

Input /mealvla/action_proposal (std_msgs/String JSON):
{"contract":"libero_relative_v1","observed_at_unix":...,"actions":[[...7 floats...]]}
Output /mealvla/action_validation: a receipt, not a robot command.
Requires the publisher and this node to share synchronized wall clocks.
"""

import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from mealvla.control import ActionQueue
from mealvla.schemas import ActionChunk


class ActionGuardNode(Node):
    def __init__(self):
        super().__init__("mealvla_action_guard")
        self.declare_parameter("ttl_s", 1.0)
        self.declare_parameter("horizon", 10)
        self.output = self.create_publisher(String, "/mealvla/action_validation", 10)
        self.subscription = self.create_subscription(String, "/mealvla/action_proposal", self.receive, 10)
        self.get_logger().info("MealVLA DRY RUN: validation only, no hardware control")

    def receive(self, message):
        try:
            body = json.loads(message.data)
            if not isinstance(body, dict) or body.get("contract") != "libero_relative_v1":
                raise ValueError("Unsupported action space")
            now = time.monotonic()
            age = time.time() - float(body["observed_at_unix"])
            chunk = ActionChunk(np.asarray(body["actions"], dtype=np.float32), now - age)
            queue = ActionQueue(self.get_parameter("horizon").value, self.get_parameter("ttl_s").value)
            queue.replace(chunk, now)
            result = {"accepted": True, "dry_run": True, "message": "valid proposal; no execution"}
        except (ValueError, TypeError, KeyError, OverflowError, TimeoutError) as exc:
            result = {"accepted": False, "dry_run": True, "error": str(exc)}
        self.output.publish(String(data=json.dumps(result)))


def main():
    rclpy.init()
    node = ActionGuardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
