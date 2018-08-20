var nj = require('numjs');

class NN {
  constructor() {
    this.inputSize = 2;
    this.outputSize = 1;
    this.hiddenSize = 3;

    this.w1 = nj.random([this.inputSize, this.hiddenSize]);
    this.w2 = nj.random([this.hiddenSize, this.outputSize]);
  }

  forward(X) {
    this.z = nj.dot(X, this.w1);
    this.z2 = this.sigmoid(this.z);
    this.z3 = nj.dot(this.z2, this.w2);
    this.o = this.sigmoid(this.z3);

    return this.o;
  }

  backward(X, y, o) {
    this.o_error = y.subtract(o);
    this.o_delta = this.o_error.multiply(this.sigmoidPrime(o));

    this.z2_error = this.o_delta.dot(this.w2.T);
    this.z2_delta = this.z2_error.multiply(this.sigmoidPrime(this.z2));

    this.w1 = this.w1.add(X.T.dot(this.z2_delta));
    this.w2 = this.w2.add(this.z2.T.dot(this.o_delta));
  }

  sigmoid(s) {
    return nj.divide(nj.ones(s.shape), nj.exp(s.negative()).add(1));
  }

  sigmoidPrime(s) {
    return nj.multiply(s, nj.subtract(nj.ones(s.shape), s));
  }

  train(X, y) {
    const o = this.forward(X);
    this.backward(X, y, o);
  }

  predict(xPredicted) {
    const z = nj.dot(xPredicted, this.w1);
    const z2 = this.sigmoid(z);
    const z3 = nj.dot(z2, this.w2);
    return this.sigmoid(z3);
  }
}

// dataset
let X = nj.array([[2, 9], [1, 5], [3, 6]], 'float32');
let y = nj.array([[92], [86], [89]], 'float32');
let xPredicted = nj.array([[4, 8]], 'float32');

// scale
X = X.divide(X.max());
xPredicted = xPredicted.divide(xPredicted.max());
y = y.divide(100);

const nn = new NN();

// train 10000 times
for (let i = 0; i < 10000; i++) {
  console.log('training #', i);
  if (i === 9999) {
    console.log('\nInput: \n' + X);
    console.log('\nActual Output: \n' + y);
    console.log('\nPredicted Output: \n' + nn.forward(X));
    console.log('\nLoss: \n' + nj.mean(nj.power(y.subtract(nn.forward(X)), 2)));
    console.log('======');
  }
  nn.train(X, y);
}

// predition
const predict = nn.predict(xPredicted);

console.log(`Predicted data based on trained weights: `);
console.log(`Input (scaled): \n` + xPredicted);
console.log(`Output: \n` + predict);
