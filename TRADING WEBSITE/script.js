let model;

// load the model when page loads
window.onload = async function() {
  model = await tf.loadLayersModel('./model/model.json');
  console.log("Model loaded!");
};

document.getElementById('upload').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file || !model) return;

  const image = await loadImage(file);
  const prediction = await predict(image);
  
  document.getElementById('result').innerText = 
    `Prediction: ${prediction.className} (${(prediction.probability * 100).toFixed(2)}%)`;
});

async function loadImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => resolve(img);
  });
}

async function predict(img) {
  // Preprocess: match your Teachable Machine model input size (default 224x224)
  const tensor = tf.browser.fromPixels(img)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .expandDims(0)
    .div(255.0);

  const predictions = await model.predict(tensor).data();
  
  // You’ll need to match these labels to your model's classes
  const labels = ['Fall', 'Rise'];
  const maxIdx = predictions.indexOf(Math.max(...predictions));

  return { className: labels[maxIdx], probability: predictions[maxIdx] };
}
