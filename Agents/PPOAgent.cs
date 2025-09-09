using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using PPO_Proj_CS.NNet;

namespace PPO_Proj_CS.Agents
{
	public class PPOAgent
	{
		private NeuralNet net;
		private double gamma = 0.99;
		private double epsilon = 0.2;
		private double lr = 0.001;                  // learning rate, adjust as needed.
		private Random rng = new Random();

		public int actionSize;

		public PPOAgent(NeuralNet net)
		{
			this.net = net;
			actionSize = net.outputSize;
		}

		// Sample continuous action
		public Vector<double> GetContinuousAction(Vector<double> state)
		{
			var (mu, std) = net.ForwardContinuous(state);
			var action = Vector<double>.Build.Dense(actionSize);
			for (int i = 0; i < actionSize; i++)
				action[i] = mu[i] + std[i] * SampleNormal();
			return action;
		}

		private double SampleNormal()
		{
			double u1 = 1.0 - rng.NextDouble();
			double u2 = 1.0 - rng.NextDouble();
			return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
		}

		// Batch update using actor-critic (PPO clipping simplified)
		public void Update(List<Vector<double>> states, List<Vector<double>> actions, List<double> rewards)
		{
			// Compute returns
			List<double> returns = new List<double>();
			double G = 0;
			for (int t = rewards.Count - 1; t >= 0; t--)
			{
				G = rewards[t] + gamma * G;
				returns.Insert(0, G);
			}

			// Compute advantages
			List<double> advantages = new List<double>();
			for (int i = 0; i < states.Count; i++)
			{
				double V = net.ForwardValue(states[i]);
				advantages.Add(returns[i] - V);
			}

			// Normalize advantages
			double mean = 0.0;
			foreach (var a in advantages) mean += a;
			mean /= advantages.Count;

			double std = 0.0;
			foreach (var a in advantages) std += (a - mean) * (a - mean);
			std = Math.Sqrt(std / advantages.Count) + 1e-8;

			for (int i = 0; i < advantages.Count; i++)
				advantages[i] = (advantages[i] - mean) / std;

			// Apply updates
			for (int i = 0; i < states.Count; i++)
			{
				var state = states[i];

				// Critic update
				net.BackpropValue(state, returns[i], lr);

				// Actor update
				var (mu, stdActor) = net.ForwardContinuous(state);
				double r = 1.0; // placeholder until old mu/std are implemented
				double clippedAdv = Math.Min(Math.Max(r, 1 - epsilon), 1 + epsilon) * advantages[i];

				net.BackpropActor(state, actions[i], clippedAdv, lr);
			}
		}

		private double GaussianPdf(double x, double mean, double std)
		{
			double var = std * std;
			return Math.Exp(-Math.Pow(x - mean, 2) / (2 * var)) / (Math.Sqrt(2 * Math.PI) * std);
		}
	}
}
