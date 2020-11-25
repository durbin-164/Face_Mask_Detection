let video;
let classifier;
let model;
let modelURL = 'tfjsv3/model.json';
let label = "waiting...";
let model_git = 'https://github.com/durbin-164/Face_Mask_Detection/blob/master/model/tfjsv3/model.json'


function setup() {
  createCanvas(640, 520);
  // Create the video
  video = createCapture(VIDEO);
  video.hide();

  console.log("call classfy")
  // model = await tf.loadLayersModel(modelURL)
  // classifier =  ml5.imageClassifier("MobileNet", video, modelReady);
  // // classifier = await tf.loadLayersModel(model_git)
  // console.log(classifier)

  const featureExtractor = ml5.featureExtractor("MobileNet", modelLoaded);

// Create a new classifier using those features and with a video element
  classifier = featureExtractor.classification(video, videoReady);

  // STEP 2.1: Start classifying
  
}

// STEP 2.2 classify!
function classifyVideo() {
    console.log("hello classifyVideo")
  classifier.classify( gotResults);
}




// When the model is loaded
function modelLoaded() {
  console.log("Model Loaded!");
  classifier.load(modelURL, customModelReady)
}


// Triggers when the video is ready
function videoReady() {
  console.log("The video is ready!");
  
}

function customModelReady(){
  console.log("custom model ready");
  classifyVideo();
}

function draw() {
  background(0);
  
  // Draw the video
  image(video, 0, 0);

  // STEP 4: Draw the label
  textSize(32);
  textAlign(CENTER, CENTER);
  fill(255);
  text(label, width / 2, height - 16);
}


// STEP 3: Get the classification!
function gotResults(error, results) {
  // Something went wrong!
  console.log("in got results")
  if (error) {
    console.error(error);
    return;
  }
  console.log(results)
  // Store the label and classify again!
  label = results[0].label;
  console.log(results[0].label);
  classifyVideo();
}