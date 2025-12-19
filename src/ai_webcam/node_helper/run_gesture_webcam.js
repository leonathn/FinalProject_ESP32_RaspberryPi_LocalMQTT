// Simple Node.js runner for the Teachable Machine gesture model.
// Captures a webcam frame every 5 seconds and logs the top prediction.

const fs = require('fs/promises');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const NodeWebcam = require('node-webcam');

const MODEL_DIR = path.join(__dirname, 'src', 'gesture_model');
const MODEL_URL = `file://${path.join(MODEL_DIR, 'model.json')}`;
const METADATA_PATH = path.join(MODEL_DIR, 'metadata.json');
const CAPTURE_MS = 5000;
const INPUT_SIZE = 224;

async function loadModelAndLabels() {
  const [model, metadataRaw] = await Promise.all([
    tf.loadLayersModel(MODEL_URL),
    fs.readFile(METADATA_PATH, 'utf8'),
  ]);
  const { labels = [] } = JSON.parse(metadataRaw);
  return { model, labels };
}

function createWebcam() {
  return NodeWebcam.create({
    width: INPUT_SIZE,
    height: INPUT_SIZE,
    delay: 0,
    saveShots: false,
    output: 'png',
    device: false,
    callbackReturn: 'buffer',
    verbose: false,
  });
}

function captureFrame(webcam) {
  return new Promise((resolve, reject) => {
    webcam.capture('gesture-frame', (err, data) => {
      if (err) return reject(err);
      resolve(data); // Buffer when callbackReturn is "buffer"
    });
  });
}

function predict(model, buffer) {
  return tf.tidy(() => {
    const image = tf.node.decodeImage(buffer, 3);
    const resized = tf.image.resizeBilinear(image, [INPUT_SIZE, INPUT_SIZE]);
    const normalized = resized.div(255);
    const batched = normalized.expandDims(0);
    const logits = model.predict(batched);
    return logits.dataSync();
  });
}

async function main() {
  const { model, labels } = await loadModelAndLabels();
  const webcam = createWebcam();
  let running = false;

  console.log('Model loaded. Capturing from webcam every 5 seconds...');

  setInterval(async () => {
    if (running) return;
    running = true;
    try {
      const frame = await captureFrame(webcam);
      const scores = predict(model, frame);
      const topIdx = scores.indexOf(Math.max(...scores));
      const label = labels[topIdx] ?? `class_${topIdx}`;
      const confidence = (scores[topIdx] * 100).toFixed(1);
      console.log(`[${new Date().toISOString()}] ${label} (${confidence}%) | scores=${scores.map(v => v.toFixed(2)).join(', ')}`);
    } catch (err) {
      console.error('Capture/predict error:', err.message || err);
    } finally {
      running = false;
    }
  }, CAPTURE_MS);
}

main().catch((err) => {
  console.error('Failed to start:', err);
});
