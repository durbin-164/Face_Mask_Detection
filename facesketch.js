
let model;
let video;
let label;
let isModelReady = false;
let facemodel ;
let start;
let size;
let rectColor;
let canvas;
let cusimage;
let img;

async function modelLoad(modelReady){
    model = await tf.loadLayersModel('model/model.json');
    facemodel = await blazeface.load();

    modelReady(); 
}

function makeImage(image){
    let imageArry = image.arraySync();

    let shape = image.shape;

    img = createImage(shape[0], shape[1]);
    img.loadPixels();

    for(let i = 0; i<shape[0]; i++){
        for(let j = 0; j<shape[1]; j++){
            let cc = imageArry[i][j];
            let p_color = color(cc[0],cc[1],cc[2],255);

            img.set(i,j, color(p_color));
        }
    }

    img.updatePixels(); 
}

function getResult(){
    video.loadPixels()

    tf.tidy(() => {
        let image = tf.browser.fromPixels(video.imageData).transpose([1,0,2])
    
        let dim1 = [int(start[0]),int(start[1]),0]
        let dim2 = [int(size[0]), int(size[1]), 3]
        try{
            image = image.slice(dim1, dim2);
            makeImage(image);

        }catch(error){}
        
        
        image = tf.image.resizeBilinear(image, [224,224]).reshape([-1,224,224,3])
        let result = model.predict(image)
        result = result.dataSync();

        if(result[0]>result[1]){
            label = "mask";
            rectColor= color(0,255,0,255);
        }else{
            label = "no mask";
            rectColor= color(255,0,0,255);
        }
    });
}

async function getFace(){
    video.loadPixels()
    const returnTensors = false;
    const predictions = await facemodel.estimateFaces(video.imageData, returnTensors);

    if(predictions.length){
        for(let i =0; i< predictions.length; i++){
            start = predictions[i].topLeft;
            let end = predictions[i].bottomRight;

            start = [max(0, start[0]), max(0, start[1])]
            end = [end[0], end[1]]

            size = [end[0]-start[0], end[1]- start[1]]
            getResult();

        }
    }
}


function modelReady(){
    console.log("model Ready")
    isModelReady = true;
}

function videoReady(){
    console.log("video ready.");
    modelLoad(modelReady);
}


function setup(){
    canvas = createCanvas(600,550)
    video = createCapture(VIDEO, videoReady);
    video.hide()
}

function draw(){
    background(0)
    image(video, 0,0)

    fill(255)
    textSize(32);
    text(label, 20,height-20);

    if (isModelReady){
        getFace();
    }

    if(start && size&& start.length>=2 && size.length >=2){
        rectMode(CORNER);
        noFill();
        stroke(rectColor); 
        rect(start[0], start[1], size[0], size[1]);
        
    }

    if(img){
        image(img, 5,5);
    }
            
}



