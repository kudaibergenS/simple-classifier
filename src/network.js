import { default as nj } from 'numjs';
import { zip, range, idxOfMax } from './utils';

class Network2 {

  constructor(sizes) {
    this.sizes = sizes;
    this.layersNum = sizes.length;
    this.weights = this._initWeights(sizes);
    this.biases = this._initBiases(sizes);
  }

  feedForward(a) {
    // z[0] - weights at n-th layer, z[1] - biases at n-th layer
    // z[1].T return the transposition of biases matrix
    for (let tupple of zip([this.biases, this.weights]))
      a = this.sigmoid(nj.dot(tupple[1], a).add(tupple[0]));

    return a;
  }

  sgd(trainingData, alpha) {
     // stochastic gradient descent
     // alpha is a learning rate
     for (let i = 0; i < 20000; i++) {
       let randomData = trainingData[Math.floor(Math.random() * trainingData.length)];

       const targetInput = nj.array([randomData.slice(0 , -1)]).T;

       let targetOutput;

       if (randomData[2] === 1)
        targetOutput = nj.array([[1], [0]])
       else
        targetOutput = nj.array([[0], [1]])

       const delta = this.backProp(targetInput, targetOutput);

       // Updating weights, tupple[0] is prev Weight, tupple[1] is delta Weight
       let zippedWeights = zip([this.weights, delta.nablaW]);
       for (let tupple of zippedWeights) {
         this.weights[zippedWeights.indexOf(tupple)] = tupple[0].subtract((tupple[1].multiply(alpha)));
       }
       // Updating biases
       let zippedBiases = zip([this.biases, delta.nablaB]);
       for (let tupple of zippedBiases) {
         this.biases[zippedBiases.indexOf(tupple)] = tupple[0].subtract((tupple[1].multiply(alpha)));
       }
     }
     console.log('mmm, feeling educated...');
   }

  backProp(x, y) {
    const nablaW = this._initNablaW(),
          nablaB = this._initNablaB();

    let activation = x,
        activations = [x],
        zs = []; // array of z's

    // feedforward
    for (let tupple of zip([this.biases, this.weights])) {
      let z = nj.dot(tupple[1], activation).add(tupple[0]);
      zs.push(z);
      activation = this.sigmoid(z);
      activations.push(activation);
    }

    this.printCost(activations[activations.length - 1], y) // prints the error cost

    let dcost_da = this.costDerivative(activations[activations.length - 1], y);
    let da_dz = this.sigmoidPrime(zs[zs.length - 1]);
    // dz_db = 1 so it is scipped
    let dz_dw = activations[activations.length - 2]
    let dcost_db = dcost_da.multiply(da_dz);
    // change in cost with respect to weights
    let dcost_dw = nj.dot(dcost_db, dz_dw.T);

    nablaW[nablaW.length - 1] = dcost_dw;
    nablaB[nablaB.length - 1] = dcost_db;

    for (let l of range(2, this.layersNum)) {
      let z = zs[zs.length - l]; // L-1 th activation vector, where L is a Last layer label
      da_dz = this.sigmoidPrime(z);

      dcost_db = nj.dot(this.weights[this.weights.length - l + 1].T, dcost_db).multiply(da_dz);
      dz_dw = activations[activations.length - l - 1];
      dcost_dw = nj.dot(dcost_db, dz_dw.T);

      nablaB[nablaB.length - l] = dcost_db;
      nablaW[nablaW.length - l] = dcost_dw;
    }
    return { nablaW, nablaB };
  }

  evaluate(data) {
    const input = nj.array([data]).T; // N x 1 input matrix
    const output = this.feedForward(input);

    console.log('============================================');
    console.log('input', input.tolist());
    console.log('output', output.tolist());
    return idxOfMax(output.T.tolist()[0]);
  }

  printCost(pred, target) {
    let err = pred.subtract(target);
    console.log('error cost:', err.multiply(err).sum());
  }

  costDerivative(pred, target) {
    // dcost_da = 2 * (pred - target)
    return pred.subtract(target).multiply(2);
  }

  sigmoid(z) {
    // matrice of ones
    const ones = nj.ones(z.shape);
    // denominator: 1 + e^(-x)
    const d = nj.exp(nj.negative(z)).add(ones)
    // returns the matrix of 1 / 1 + e^(-x)
    return ones.divide(d);
  }

  sigmoidPrime(z) {
    const ones = nj.ones(z.shape);
    // sigm(z) * (1 - sigm(z))
    return this.sigmoid(z).multiply(ones.subtract(this.sigmoid(z)));
  }

  _initWeights(sizes) {
    const weights = [];

    for (let z of zip([sizes.slice(0, -1), sizes.slice(1)]))
      weights.push(nj.random([z[1], z[0]]))

    return weights;
  }

  _initBiases(sizes) {
    const biases = [];

    for (let s of sizes.slice(1))
      biases.push(nj.random([s, 1]));

    return biases;
  }

  _initNablaB() {
    const nablaB = [];

    for (let b of this.biases)
      nablaB.push(nj.zeros(b.shape));

    return nablaB;
  }

  _initNablaW() {
    const nablaW = [];

    for (let w of this.weights)
      nablaW.push(nj.zeros(w.shape));

    return nablaW;
  }

}

export default Network2;
