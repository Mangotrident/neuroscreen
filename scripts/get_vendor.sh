#!/usr/bin/env bash
# Downloads Google's MediaPipe Face Mesh and Pose web packages (Apache-2.0) from npm into web/vendor,
# so the demo and the robot can run fully offline. Without them, dist/demo.html loads the same files from jsDelivr.
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
( cd "$tmp" && npm pack @mediapipe/face_mesh@0.4.1633559619 @mediapipe/pose@0.5.1675469404 >/dev/null )
mkdir -p web/vendor/face_mesh web/vendor/pose
tar -xzf "$tmp"/mediapipe-face_mesh-*.tgz -C "$tmp" && cp "$tmp"/package/* web/vendor/face_mesh/ && rm -rf "$tmp"/package
tar -xzf "$tmp"/mediapipe-pose-*.tgz -C "$tmp" && cp "$tmp"/package/* web/vendor/pose/
rm -f web/vendor/pose/pose_landmark_full.tflite web/vendor/pose/pose_landmark_heavy.tflite   # the lite model is used
rm -rf "$tmp"; echo "MediaPipe models in web/vendor"
