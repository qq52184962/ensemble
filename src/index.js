import { matrix, index as matrixIndex } from 'mathjs';

class DecisionTree {
  constructor (maxDepth) {
    this.maxDepth = maxDepth || 3;
  }

  _entropyFunc (num, total) {
    return -num / total * Math.log2(num / total);
  }

  _calcEntropy (classA, classB) {
    if (!classA || !classB) {
      return 0;
    }
    const total = classA + classB;
    return this._entropyFunc(classA, total) + this._entropyFunc(classB, total);
  }

  getEntropy (predict, real) {
    if (!Array.isArray(predict) || !Array.isArray(real)) {
      throw Error('Expect to get array.');
    }
    if (predict.length !== real.length) {
      throw Error('Array length is inconsistent.');
    }
    const m = {
      aTrue: 0,
      aFalse: 0,
      bTrue: 0,
      bFalse: 0
    };
    predict.forEach((c, i) => {
      if (c) {
        if (real[i]) {
          m.aTrue++;
        } else {
          m.aFalse++;
        }
      } else {
        if (real[i]) {
          m.bTrue++;
        } else {
          m.bFalse++;
        }
      }
    });
    const a = m.aTrue / (m.aTrue + m.aFalse) * this._calcEntropy(m.aTrue, m.aFalse);
    const b = m.bTrue / (m.bTrue + m.bFalse) * this._calcEntropy(m.bTrue, m.bFalse);
    return a + b;
  }

  _bestSplit (col, y) {
    let entropy = 10;
    let cutoff;
    y = y.clone();
    const [r] = col.size();
    col.forEach(val => {
      const predict = col.map(c => c < val).reshape([r]).toArray();
      const score = this.getEntropy(predict, y.reshape([r]).toArray());
      if (score < entropy) {
        entropy = score;
        cutoff = val;
      }
    });
    return { entropy, cutoff };
  }

  _bestSplitForAll (x, y) {
    const [row, col] = x.size();
    let entropy = 10;
    let cutoff;
    let index;
    for (let i = 0; i < col; i++) {
      const column = x.subset(matrixIndex([...new Array(row).keys()], i));
      const { entropy: score, cutoff: currCutoff } = this._bestSplit(column, y);
      if (score === 0) {
        return { i, score, currCutoff };
      } else if (score < entropy) {
        entropy = score;
        cutoff = currCutoff;
        index = i;
      }
    }
    return { index, entropy, cutoff };
  }

  _fit (x, y, depth) {
    if (depth >= this.maxDepth) {
      return null;
    }
    const [r, c] = x.size();
    if (!r) {
      return null;
    }
    const { index, entropy, cutoff } = this._bestSplitForAll(x, y);
    if (!cutoff) {
      return null;
    }
    const xCol = x.subset(matrixIndex([...new Array(r).keys()], index));
    const left = xCol.map(w => w < cutoff).reshape([r]).toArray();
    // create left & right index array
    const leftIndices = [], rightIndices = [];
    for (let i = 0; i < left.length; i++) {
      if (left[i]) leftIndices.push(i);
      else rightIndices.push(i);
    }
    const tNode = {
      indexCol: index,
      entropy,
      cutoff,
      left: this._fit(
        x.subset(matrixIndex(leftIndices, [...new Array(c).keys()])), 
        y.subset(matrixIndex(leftIndices, 0)), 
        depth + 1
      ),
      right: this._fit(
        x.subset(matrixIndex(rightIndices, [...new Array(c).keys()])), 
        y.subset(matrixIndex(rightIndices, 0)), 
        depth + 1
      )
    };
    return tNode;
  }

  fit (x, y) {
    return this._fit(x, y, 0);
  }

  predict () {

  }
};

const tree = new DecisionTree();
const data = matrix([
  [1,2,3,true],
  [0,1,0,false],
  [2,3,1,true],
  [2,1,2,false],
  [1,2,0,false],
  [3,2,1,true],
  [0,0,0,false],
  [2,3,3,true],
  [3,2,2,true]
]);
const [r, c] = data.size();
const x = data.subset(matrixIndex([...new Array(r).keys()], [...new Array(c - 1).keys()]));
const y = data.subset(matrixIndex([...new Array(r).keys()], c - 1));
console.log(tree.fit(x, y));