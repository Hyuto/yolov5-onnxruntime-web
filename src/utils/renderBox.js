import labels from "./labels.json";

/**
 * Render prediction boxes
 * @param {React.MutableRefObject} canvasRef canvas tag reference
 * @param {number} threshold threshold number
 * @param {Array} boxes_data boxes array
 * @param {Array} scores_data scores array
 * @param {Array} classes_data class array
 */
export const renderBoxes = (canvasRef, boxes_data, scores_data, classes_data) => {
  const ctx = canvasRef.current.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  // font configs
  const font = "18px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";

  for (let i = 0; i < scores_data.length; ++i) {
    const klass = labels[classes_data[i]];
    const score = (scores_data[i] * 100).toFixed(1);
    const [x1, y1, width, height] = boxes_data[i];

    // Draw the bounding box.
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, width, height);

    // Draw the label background.
    ctx.fillStyle = "#00FF00";
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10); // base 10
    ctx.fillRect(x1 - 1, y1 - (textHeight + 2), textWidth + 2, textHeight + 2);

    // Draw labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(klass + " - " + score + "%", x1 - 1, y1 - (textHeight + 2));
  }
};
