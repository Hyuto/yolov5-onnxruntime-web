const nms = (foundLocations, overlapThresh) => {
  if (foundLocations.length === 0) {
    return [];
  }

  const pick = [];

  foundLocations = foundLocations.map((box) => {
    return {
      x1: box[0],
      y1: box[1],
      x2: box[0] + box[2],
      y2: box[1] + box[3],
      width: box[2],
      height: box[3],
      area: (box[1] + 1) * (box[0] + 1),
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

const argsort = (arr) => {
  let decor = (v, i) => [v, i];
  let undecor = (a) => a[1];
  return arr.map(decor).sort().map(undecor);
};

/**
 * Non Max Suppression algorithm
 * implemented from erceth/non-maximum-suppression : https://github.com/erceth/non-maximum-suppression
 * @param {Array} foundLocations founded boxes
 * @param {Number} overlapThresh overlap threshold
 * @returns {Array} selected boxes
 */
const NMSFast = (foundLocations, overlapThresh) => {
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

export { nms, NMSFast };
