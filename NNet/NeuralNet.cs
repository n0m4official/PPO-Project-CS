using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace PPO_Proj_CS.NNet
{
	public class NeuralNet
	{
		// Actor
		public int inputSize, hiddenSize, outputSize;
		public Matrix<double> W1, W2;
		public Vector<double> b1, b2;
		public Vector<double> logStd; // std deviation for Gaussian policy

		// Critic
		public Matrix<double> V_W1, V_W2;
		public Vector<double> V_b1, V_b2;

		private Random rng = new Random();

		public NeuralNet(int inputSize, int hiddenSize, int outputSize)
		{
			this.inputSize = inputSize;
			this.hiddenSize = hiddenSize;
			this.outputSize = outputSize;

			// Actor
			W1 = Matrix<double>.Build.Random(hiddenSize, inputSize, new Normal(0, 0.1));
			b1 = Vector<double>.Build.Dense(hiddenSize, 0);
			W2 = Matrix<double>.Build.Random(outputSize, hiddenSize, new Normal(0, 0.1));
			b2 = Vector<double>.Build.Dense(outputSize, 0);
			logStd = Vector<double>.Build.Dense(outputSize, -0.5);

			// Critic
			V_W1 = Matrix<double>.Build.Random(hiddenSize, inputSize, new Normal(0, 0.1));
			V_b1 = Vector<double>.Build.Dense(hiddenSize, 0);
			V_W2 = Matrix<double>.Build.Random(1, hiddenSize, new Normal(0, 0.1));
			V_b2 = Vector<double>.Build.Dense(1, 0);
		}

		// Actor forward: returns mean and std of Gaussian
		public (Vector<double> mu, Vector<double> std) ForwardContinuous(Vector<double> x)
		{
			var a1 = ReLU(W1 * x + b1);
			var mu = W2 * a1 + b2;
			var std = logStd.Map(Math.Exp);
			return (mu, std);
		}

		// Critic forward
		public double ForwardValue(Vector<double> x)
		{
			var a1 = ReLU(V_W1 * x + V_b1);
			return (V_W2 * a1 + V_b2)[0];
		}

		private Vector<double> ReLU(Vector<double> v) => v.Map(e => Math.Max(0, e));

		// Critic backprop
		// NeuralNet.cs
		public void BackpropActor(Vector<double> x, Vector<double> action, double advantage, double lr)
		{
			var (mu, std) = ForwardContinuous(x);

			// d(mu) = (action - mu) / std^2 * advantage
			var delta = Vector<double>.Build.Dense(action.Count);
			for (int i = 0; i < action.Count; i++)
				delta[i] = (action[i] - mu[i]) / (std[i] * std[i]) * advantage;

			// Gradients for W2 and b2
			var z1 = W1 * x + b1;
			var a1 = z1.Map(e => Math.Max(0, e));
			var dW2 = delta.ToColumnMatrix() * a1.ToRowMatrix();
			var db2 = delta;

			// Backprop to hidden layer
			var delta1 = (W2.TransposeThisAndMultiply(delta)).PointwiseMultiply(z1.Map(e => e > 0 ? 1.0 : 0.0));
			var dW1 = delta1.ToColumnMatrix() * x.ToRowMatrix();
			var db1 = delta1;

			W2 -= lr * dW2;
			b2 -= lr * db2;
			W1 -= lr * dW1;
			b1 -= lr * db1;

			// Optionally update logStd with gradient (delta * (action - mu)^2 - 1)
		}

		// Critic backprop
		public void BackpropValue(Vector<double> x, double target, double lr)
		{
			var z1 = V_W1 * x + V_b1;
			var a1 = ReLU(z1);
			double v = (V_W2 * a1 + V_b2)[0];
			double delta = v - target;

			var dW2 = delta * a1.ToRowMatrix();
			var db2 = Vector<double>.Build.Dense(1, delta);
			var delta1 = (V_W2.TransposeThisAndMultiply(Vector<double>.Build.Dense(1, delta)))
							.PointwiseMultiply(z1.Map(e => e > 0 ? 1.0 : 0.0));
			var dW1 = delta1.ToColumnMatrix() * x.ToRowMatrix();
			var db1 = delta1;

			V_W2 -= lr * dW2;
			V_b2 -= lr * db2;
			V_W1 -= lr * dW1;
			V_b1 -= lr * db1;
		}
	}
}
