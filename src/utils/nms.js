/**
 * Perform Non Maximum Suppression to filter overlapping boxes
 * @param {Array[Object]} boxes boxes
 * @param {Number} overlapThresh overlapping threshold
 * @returns {Array[Object]} boxes
 */
export const NMS = (boxes, overlapThresh) => {
  if (boxes.length === 0) {
    return [];
  }

  const pick = [];

  boxes.sort((b1, b2) => {
    return b1.confidence - b2.confidence;
  });

  while (boxes.length > 0) {
    let last = boxes[boxes.length - 1];
    pick.push(last);
    let suppress = [last];

    for (let i = 0; i < boxes.length - 1; i++) {
      const box = boxes[i];
      const xx1 = Math.max(box.bounding[0], last.bounding[0]);
      const yy1 = Math.max(box.bounding[1], last.bounding[1]);
      const xx2 = Math.min(box.bounding[0] + box.bounding[2], last.bounding[0] + last.bounding[2]);
      const yy2 = Math.min(box.bounding[1] + box.bounding[3], last.bounding[1] + last.bounding[3]);
      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const overlap = (w * h) / ((box.bounding[2] + 1) * (box.bounding[3] + 1));

      if (overlap > overlapThresh) {
        suppress.push(boxes[i]);
      }
    }

    boxes = boxes.filter((box) => {
      return !suppress.find((supp) => {
        return supp === box;
      });
    });
  }
  return pick;
};
