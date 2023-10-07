let canvas,ctx;
let mouseX,mouseY,mouseDown=0;
let touchX,touchY;
let model;

function init() {
    canvas = document.getElementById('sketchpad');
    
    ctx = canvas.getContext('2d');
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.width,canvas.height);
 
    if (ctx) {
        canvas.addEventListener('mousedown', sketchpad_mouseDown, false);          
        canvas.addEventListener('mousemove', sketchpad_mouseMove, false);          
        window.addEventListener('mouseup', sketchpad_mouseUp, false);           
        canvas.addEventListener('touchstart', sketchpad_touchStart,false);
        canvas.addEventListener('touchmove', sketchpad_touchMove, false); 
    }
}

function draw(ctx,x,y,isDown) {
    lastX = x; 
    lastY = y; 

    if (isDown) {   
        ctx.beginPath();
        ctx.strokeStyle = "white";
        ctx.lineWidth = '15';       
        ctx.lineJoin = ctx.lineCap = 'round';   
        ctx.moveTo(lastX, lastY);      
        ctx.lineTo(x,y);
        ctx.closePath();   
        ctx.stroke();    
    }   
}


function sketchpad_mouseDown() {
    mouseDown=1;    
    draw(ctx,mouseX,mouseY, false );
}

function sketchpad_mouseUp() {    
    mouseDown=0;
}

function getMousePos(e) {        
    if (e.offsetX) {        
      mouseX = e.offsetX;        
      mouseY = e.offsetY;    
    }    
    else if (e.layerX) {        
      mouseX = e.layerX;        
      mouseY = e.layerY;    
    } 
}

function sketchpad_mouseMove(e) {
    getMousePos(e);
    if (mouseDown==1) {
        draw(ctx,mouseX,mouseY, true);
    }
}

function getTouchPos(e) {    
    
    if(e.touches) {   
   
      if (e.touches.length == 1) {            
        var touch = e.touches[0];            
        touchX=touch.pageX-touch.target.offsetLeft;               
        touchY=touch.pageY-touch.target.offsetTop;        
      }
    }
}

function sketchpad_touchStart(e) {     
    getTouchPos(e);    
    draw(ctx,touchX,touchY, false);    
}


function sketchpad_touchMove(e) {     
    getTouchPos(e);    
    draw(ctx,touchX,touchY,true);    
}

document.getElementById('clear_button').addEventListener("click",  
    function(){  
        ctx.clearRect(0, 0, canvas.width, canvas.height);  
        ctx.fillStyle = "black"; 
        ctx.fillRect(0, 0, canvas.width, canvas.height);
});

(async function(){  
    console.log("model loading...");  
    model = await tf.loadLayersModel("https://github.com/fvandu1/capstone/resources/model/model.json");
    console.log("model loaded..");
})();


function preprocessCanvas(image) { 
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([28, 28]).mean(2).expandDims(2).expandDims().toFloat(); 
    console.log(tensor.shape); 
    return tensor.div(255.0);
}

document.getElementById('predict_button').addEventListener("click",async function(){       
    let tensor = preprocessCanvas(canvas); 
    console.log(tensor)   
    let predictions = await model.predict(tensor).data();  
    console.log(predictions)  
    let results = Array.from(predictions);    
    displayLabel(results);    
    console.log(results);
});

function displayLabel(data) { 
    let max = data[0];    
    let maxIndex = 0;     
    for (let i = 1; i < data.length; i++) {        
      if (data[i] > max) {            
        maxIndex = i;            
        max = data[i];        
      }
    }
    document.getElementById('result').innerHTML = maxIndex;  
    document.getElementById('confidence').innerHTML = "Confidence: "+(max*100).toFixed(2) + "%";
}