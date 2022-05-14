export const nms = (foundLocations, scores, overlapThresh) => {
  if (foundLocations.length === 0) {
    return [];
  }

  const pick = [];

  foundLocations = foundLocations.map((box, index) => {
    return {
      x1: box[0],
      y1: box[1],
      x2: box[0] + box[2],
      y2: box[1] + box[3],
      width: box[2],
      height: box[3],
      area: (box[1] + 1) * (box[0] + 1),
      score: scores[index],
    };
  });

  foundLocations.sort((b1, b2) => {
    return b1.score - b2.score;
  });

  while (foundLocations.length > 0) {
    let last = foundLocations[foundLocations.length - 1];
    pick.push(last);
    let suppress = [last];

    for (let i = 0; i < foundLocations.length - 1; i++) {
      const box = foundLocations[i];
      const xx1 = Math.max(box.x1, last.x1);
      const yy1 = Math.max(box.y1, last.y1);
      const xx2 = Math.min(box.x2, last.x2);
      const yy2 = Math.min(box.y2, last.y2);
      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const overlap = (w * h) / box.area;
      if (overlap > overlapThresh) {
        suppress.push(foundLocations[i]);
      }
    }

    foundLocations = foundLocations.filter((box) => {
      return !suppress.find((supp) => {
        return supp === box;
      });
    });
  }
  return pick;
};

const argsort = (arr, ascending = true) => {
  let decor = (v, i) => [v, i];
  let undecor = (a) => a[1];
  if (ascending) return arr.map(decor).sort().map(undecor);
  return arr
    .map(decor)
    .sort((a, b) => b - a)
    .map(undecor);
};

/**
 * Non Max Suppression algorithm
 * implemented from erceth/non-maximum-suppression : https://github.com/erceth/non-maximum-suppression
 * @param {Array} foundLocations founded boxes
 * @param {Number} overlapThresh overlap threshold
 * @returns {Array} selected boxes
 */
export const NMSFast = (foundLocations, scores, overlapThresh) => {
  if (foundLocations.length === 0) return [];

  const pick = [],
    x1 = [],
    x2 = [],
    y1 = [],
    y2 = [],
    area = [];

  foundLocations.forEach((box) => {
    x1.push(box[0]);
    y1.push(box[1]);
    x2.push(box[0] + box[2]);
    y2.push(box[1] + box[3]);
    area.push((box[2] + 1) * (box[3] + 1));
  });

  let idxs = argsort(y2);

  while (idxs.length > 0) {
    let last = idxs.length - 1;
    let i = idxs[last];
    pick.push(i);

    const xx1 = Math.max(x1[i], ...idxs.slice(0, last).map((e) => x1[e])),
      yy1 = Math.max(y1[i], ...idxs.slice(0, last).map((e) => y1[e])),
      xx2 = Math.min(x2[i], ...idxs.slice(0, last).map((e) => x2[e])),
      yy2 = Math.min(y2[i], ...idxs.slice(0, last).map((e) => y2[e]));

    const w = Math.max(0, xx2 - xx1 + 1),
      h = Math.max(0, yy2 - yy1 + 1);
    const overlap = idxs.slice(0, last).map((e) => (w * h) / area[e]);

    idxs = idxs.filter((e, index) => {
      if (index === last || overlap[index] > overlapThresh) return false;
      return true;
    });
  }
  return pick;
};

function IOU(box1, box2) {
  let s1 = box1[2] * box1[3];
  let s2 = box2[2] * box2[3];

  let left1 = box1[0];
  let right1 = left1 + box1[2];
  let left2 = box2[0];
  let right2 = left2 + box2[2];
  let overlapW = calOverlap([left1, right1], [left2, right2]);

  let top1 = box2[1];
  let bottom1 = top1 + box1[3];
  let top2 = box2[1];
  let bottom2 = top2 + box2[3];
  let overlapH = calOverlap([top1, bottom1], [top2, bottom2]);

  let overlapS = overlapW * overlapH;
  return overlapS / (s1 + s2 + overlapS);
}

// Calculate the overlap range of two vector
function calOverlap(range1, range2) {
  let min1 = range1[0];
  let max1 = range1[1];
  let min2 = range2[0];
  let max2 = range2[1];

  if (min2 > min1 && min2 < max1) {
    return max1 - min2;
  } else if (max2 > min1 && max2 < max1) {
    return max2 - min1;
  } else {
    return 0;
  }
}

const NMSOpenCV = (boxes, scores, nmsThreshold) => {
  let zipped = [];
  for (let i = 0; i < scores.length; i++) {
    zipped.push([scores[i], boxes[i], i]);
  }

  // sort by score
  const sorted = zipped.sort((a, b) => b[0] - a[0]);
  const selected = [];
  sorted.forEach((box) => {
    let toAdd = true;
    selected.forEach((element) => {
      const iou = IOU(box[1], element[1]);
      if (iou > nmsThreshold) {
        toAdd = false;
      }
    });

    if (toAdd) {
      selected.push(box);
    }
  });

  return selected;
};
