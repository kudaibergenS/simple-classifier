import express from 'express';

import Network from './network';

const app = express();

const network = new Network([2, 4, 3, 2]);

app.listen(3012, () => {
  console.log('api started...')

  network.sgd(trainingData, 2);

  for (let i = 0; i < 72; i++) {
    console.log('pred:',
      network.evaluate(trainingData[i].slice(0, -1)) === 1 ? 'blue flower' : 'red flower',
      '||',
      'target:',
      trainingData[i][2] === 0 ? 'blue flower' : 'red flower') // 1
  }
});

const trainingData = [
  [3, 1.5, 1],
  [2, 1, 0],
  [4, 1.5, 1],
  [2.8, 1, 0],
  [3.5, 0.5, 1],
  [2, 0.5, 0],
  [5.5, 1, 1],
  [1, 1, 0],

  [3.1, 1.9, 1],
  [2.7, 1.2, 0],
  [4.1, 1.9, 1],
  [2.24, 2.12, 0],
  [3.3, 2.5, 1],
  [2.1, 1.5, 0],
  [5.2, 1.2, 1],
  [1.2, 1.9, 0],

  [5.2, 2, 1],
  [2.1, 2.4, 0],
  [4, 2.5, 1],
  [2.8, 1, 0],
  [3.5, 0.5, 1],
  [2, 0.5, 0],
  [5.5, 1, 1],
  [1, 1, 0],

  [3.9, 2.9, 1],
  [1.8, 1, 0],
  [4.45, 2.2, 1],
  [2.56, 2.42, 0],
  [3.9, 2.5, 1],
  [2.1, 1.5, 0],
  [3.22, 1.2, 1],
  [1.2, 1.9, 0],

  [4.21, 2.1, 1],
  [2.1, 2.3, 0],
  [4.2, 2.7, 1],
  [1.8, 1, 0],
  [3.4, 0.8, 1],
  [2.1, 0.6, 0],
  [4.73, 2.1, 1],
  [1.2, 1.1, 0],

  [3.9, 1.9, 1],
  [1.8, 1.2, 0],
  [3.24, 1.45, 1],
  [1.6, 1.7, 0],
  [3.52, 1.59, 1],
  [1.34, 1.45, 0],
  [3.12, 1.32, 1],
  [1.64, 1.32, 0],

  [3.12, 1.81, 1],
  [1.42, 1.3, 0],
  [3.67, 1.42, 1],
  [1.78, 1.63, 0],
  [3.68, 1.42, 1],
  [1.51, 1.6, 0],
  [3.19, 1.38, 1],
  [2.0, 1.21, 0],

  [4.79, 1.78, 1],
  [2.68, 1, 0],
  [4.42, 1.75, 1],
  [2.56, 1.42, 0],
  [4.65, 1.35, 1],
  [2.45, 1.0, 0],
  [4.12, 1.22, 1],
  [2.32, 1.53, 0],

  [4.1, 1.51, 1],
  [2.43, 0.43, 0],
  [4.58, 1.95, 1],
  [2.68, 1, 0],
  [4.48, 1.24, 1],
  [2.1, 0.6, 0],
  [4.73, 1.11, 1],
  [2.2, 1.1, 0],
];