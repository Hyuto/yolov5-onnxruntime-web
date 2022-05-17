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

export function IOU(bounding1, bounding2) {
  let s1 = bounding1[2] * bounding1[3];
  let s2 = bounding2[2] * bounding2[3];

  let left1 = bounding1[0];
  let right1 = left1 + bounding1[2];
  let left2 = bounding2[0];
  let right2 = left2 + bounding2[2];
  let overlapW = calOverlap([left1, right1], [left2, right2]);

  let top1 = bounding2[1];
  let bottom1 = top1 + bounding1[3];
  let top2 = bounding2[1];
  let bottom2 = top2 + bounding2[3];
  let overlapH = calOverlap([top1, bottom1], [top2, bottom2]);

  let overlapS = overlapW * overlapH;
  return overlapS / (s1 + s2 + overlapS);
}

export const NMS = (foundLocations, overlapThresh) => {
  if (foundLocations.length === 0) {
    return [];
  }

  const pick = [];

  foundLocations = foundLocations.map((box) => {
    // TODO: replace with vectorization
    return {
      x1: box[0],
      y1: box[1],
      x2: box[0] + box[2],
      y2: box[1] + box[3],
      width: box[2],
      height: box[3],
      area: (box[2] + 1) * (box[3] + 1),
    };
  });

  foundLocations.sort((b1, b2) => {
    return b1.y2 - b2.y2;
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
      // Test it!
      //IOU([box.x1, box.y1, box.width, box.height], [last.x1, last.y1, last.width, last.height])

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
