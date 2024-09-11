/**
 * Canvas HTML element.
 */
const canvas = document.getElementById("can");

/**
 * Canvas context.
 */
const ctx = canvas.getContext("2d");

/**
 * HTML img element to display the preprocessed image from the server.
 */
const processedImage = document.getElementById("processed-image");

/**
 * HTML div element to hold the text describing the prediction.
 */
const predictionText = document.getElementById("prediction-text");

/**
 * HTML span element to hold the predicted number.
 */
const predictionNumber = document.getElementById("prediction-number");

/**
 * HTML span element to hold the confidence percentage.
 */
const confidenceNumber = document.getElementById("confidence-number");

/**
 * Flag to set if user is currently drawing to canvas.
 */
var isDrawing = false;

/**
 * Previous x-coordinate for drawing.
 */
var prevX = 0;

/**
 * Previous y-coordinate for drawing.
 */
var prevY = 0;

/**
 * Current x-coordinate for drawing.
 */
var currX = 0;

/**
 * Current y-coordinate for drawing.
 */
var currY = 0;

const brushColor = "white";
const brushThickness = 10;
const backgroundColor = "black";

/**
 * Clears the canvas of all drawings and hides the prediction text and image.
 */
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    processedImage.src = ""
    processedImage.style.display = "none";

    predictionText.style.display = "none";
    predictionNumber.innerHTML = "";
    confidenceNumber.innerHTML = "";
}

/**
 * Draws a line to the canvas from (prevX, prevY) to (currX, currY).
 */
function drawLineToCanvas() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushThickness;
    ctx.stroke();
    ctx.closePath();
}

/**
 * Draws a dot to the canvas at (currX, currY).
 */
function drawDotToCanvas() {
    ctx.beginPath();
    ctx.fillStyle = brushColor;
    ctx.arc(currX, currY, brushThickness / 2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.closePath();
}

/**
 * Convert a data URL of an image to a blob.
 *
 * @param {string} dataURL The data URL of the image.
 *
 * @returns {Blob} The blob of the image.
 */
function dataURLToBlob(dataURL) {
    // Decode base64 string
    const byteString = atob(dataURL.split(",")[1]);
    // Extract MIME type
    const mimeString = dataURL.split(",")[0].split(":")[1].split(";")[0];

    // Write the bytes of the string to a typed array
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uint8Array = new Uint8Array(arrayBuffer);
    for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i);
    }

    return new Blob([uint8Array], { type: mimeString });
}

/**
 * Saves the canvas drawing as an image and sends it to the server for
 * prediction. If successful, the prediction, its confidence, and the
 * preprocessed image from the server are all displayed.
 */
async function saveAndSendCanvasImage() {
    // Get canvas as PNG
    const dataURL = canvas.toDataURL("image/png");

    // Convert canvas PNG URL to blob
    const blob = dataURLToBlob(dataURL);

    // Make form data with the canvas image to send to server
    const formData = new FormData();
    formData.append("file", blob, "digit.png");

    try {
        // Send request to server
        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );

        // Handle the server's response
        const result = await response.json();

        // Display the predicted number and confidence.
        predictionNumber.innerHTML = result.digit;
        confidenceNumber.innerHTML = Math.round(parseFloat(result.confidence), 2);
        predictionText.style.display = "block";

        // Display the processed image from the server
        processedImage.src = `data:image/png;base64,${result.processed_image}`;
        processedImage.style.display = "block";
    } catch (error) {
        console.error("Error: ", error);
        alert("An error occurred while sending the image.");
    }
}

/**
 * Updates previous and current coordinates and decides whether or not to draw
 * to the canvas.
 *
 * @param {"down" | "move" | "up" | "out"} eventType The type of event.
 * @param {number} clientX The x-coordinate of the event.
 * @param {number} clientY The y-coordinate of the event.
 */
function handleCanvasEvent(eventType, clientX, clientY) {
    if (eventType == "down") {
        // Update previous and current coordinates
        prevX = currX;
        prevY = currY;
        currX = clientX - canvas.offsetLeft;
        currY = clientY - canvas.offsetTop + window.scrollY;

        // Set drawing flag
        isDrawing = true;

        // Draw dot to canvas
        drawDotToCanvas();
    } else if (eventType == "up" || eventType == "out") {
        // Unset drawing flag
        isDrawing = false;
    } else  if (eventType == "move") {
        // Only draw to canvas if we are already drawing (down event was recently
        // received)
        if (isDrawing) {
            // Update previous and current coordinates
            prevX = currX;
            prevY = currY;
            currX = clientX - canvas.offsetLeft;
            currY = clientY - canvas.offsetTop + window.scrollY;

            // Draw to canvas by drawing a line and a dot at the end of the line
            drawLineToCanvas();
            drawDotToCanvas();
        }
    } else {
        console.error(`Unknown event type ${eventType}`);
    }
}

/**
 * Handles a canvas event for computer mouse (non touch screens).
 *
 * @param {"down" | "move" | "up" | "out"} eventType The type of event.
 * @param {MouseEvent} event The mouse event.
 */
function handleCanvasMouseEvent(eventType, event) {
    // Handle canvas event with the x- and y-coordinates from the event
    handleCanvasEvent(eventType, event.clientX, event.clientY);
}

/**
 * Handles a canvas event for touch screens.
 *
 * @param {"down" | "move" | "up" | "out"} eventType The type of event.
 * @param {TouchEvent} event The touch event.
 */
function handleCanvasTouchEvent(eventType, event) {
    // Prevent scrolling
    event.preventDefault();

    // Get touch
    const touch = event.touches[0];

    // Handle event
    handleCanvasEvent(eventType, touch.clientX, touch.clientY);
}

/*
 * Event listeners for mouse events.
 */

canvas.addEventListener("mousemove", (e) => {
    handleCanvasMouseEvent("move", e);
}, false);

canvas.addEventListener("mousedown", (e) => {
    handleCanvasMouseEvent("down", e);
}, false);

canvas.addEventListener("mouseup", (e) => {
    handleCanvasMouseEvent("up", e);
}, false);

canvas.addEventListener("mouseout", (e) => {
    handleCanvasMouseEvent("out", e);
}, false);

/*
 * Event listeners for touch events.
 */

canvas.addEventListener("touchmove", (e) => {
    handleCanvasTouchEvent("move", e);
}, false);

canvas.addEventListener("touchstart", (e) => {
    handleCanvasTouchEvent("down", e);
}, false);

canvas.addEventListener("touchend", (e) => {
    handleCanvasTouchEvent("up", e);
}, false);

// Clear canvas on page load
window.addEventListener("load", clearCanvas);

/*
 * Event listeners for save and clear canvas buttons.
 */

const saveCanvasButton = document.getElementById("save-canvas-button");
const clearCanvasButton = document.getElementById("clear-canvas-button");

saveCanvasButton.addEventListener("click", saveAndSendCanvasImage);
clearCanvasButton.addEventListener("click", clearCanvas);
